import pandas as pd
import torch
from torch_geometric.data import HeteroData
import os
import math
from tqdm import tqdm


def frame_tensor_to_df(frame_tensor, column_names):
    np_array = frame_tensor.cpu().numpy()
    df = pd.DataFrame([np_array], columns=column_names)
    return df


def get_global_dist_min_max(dataset):
    all_dists = []
    for sample in dataset:
        cond = sample["condition"]
        cols = sample["condition_columns"]
        for t in range(cond.shape[0]):
            for col in cols:
                if col.endswith("_dist"):
                    val = cond[t, cols.index(col)].item()
                    all_dists.append(val)
    all_dists_tensor = torch.tensor(all_dists)
    return all_dists_tensor.min().item(), all_dists_tensor.max().item()


def extract_node_features(condition_tensor, condition_columns):
    column_index_map = {col: idx for idx, col in enumerate(condition_columns)}
    device = condition_tensor.device
    dtype  = condition_tensor.dtype
    player_bases = sorted(list(set(col.rsplit("_", 1)[0] for col in condition_columns if "ball" not in col)))
    attk_bases = [base for base in player_bases if "Attk" in base or any(f"{base}_position" in col for col in condition_columns[:22*7])][:11]
    def_bases = [base for base in player_bases if base not in attk_bases][:11]

    unified_feats = []

    def get_feat(base, node_type_idx):
        feats = []
        for feat in ["x", "y", "vx", "vy"]:
            col = f"{base}_{feat}"
            if col in column_index_map:
                val = condition_tensor[column_index_map[col]]
            else:
                val = torch.tensor(0.0, device=device, dtype=dtype)
            feats.append(val)
        col = f"{base}_dist"
        if col in column_index_map:
            feats.append(condition_tensor[column_index_map[col]])
        else:
            feats.append(torch.tensor(-1.0, device=device, dtype=dtype))
        col = f"{base}_position"
        if col in column_index_map:
            feats.append(condition_tensor[column_index_map[col]])
        else:
            feats.append(torch.tensor(-1.0, device=device, dtype=dtype))
        col = f"{base}_starter"
        if col in column_index_map:
            feats.append(condition_tensor[column_index_map[col]])
        else:
            feats.append(torch.tensor(-1.0, device=device, dtype=dtype))
        feats.append(torch.tensor(float(node_type_idx), device=device, dtype=dtype))
        return torch.stack(feats)

    for base in attk_bases:
        unified_feats.append(get_feat(base, 0))
    for base in def_bases:
        unified_feats.append(get_feat(base, 1))

    # Ball
    ball_feats = []
    for feat in ["x", "y", "vx", "vy"]:
        col = f"ball_{feat}"
        val = condition_tensor[column_index_map[col]] if col in column_index_map else torch.tensor(0.0)
        ball_feats.append(val)
    ball_feats += [torch.tensor(-1.0, device=device, dtype=dtype)] * 3  # dist, position, starter
    ball_feats.append(torch.tensor(2.0, device=device, dtype=dtype))    # node_type
    unified_feats.append(torch.stack(ball_feats))

    return {"Node": torch.stack(unified_feats)}

# Edge
def build_edges_based_on_interactions(node_features):
    edge_index_dict = {}
    edge_attr_dict = {}

    all_nodes = node_features["Node"]
    node_type = all_nodes[:, -1]
    type_masks = {
        0: (node_type == 0),  # Attk
        1: (node_type == 1),  # Def
        2: (node_type == 2),  # Ball
    }

    field_x_half = 52.5
    field_y_half = 34.0
    x_scale = 1.0 / field_x_half
    y_scale = 1.0 / field_y_half
    weight_threshold = 20
    normalized_threshold = (weight_threshold**2 * x_scale**2 + weight_threshold**2 * y_scale**2)**0.5

    def add_edges(src_type_id, dst_type_id, relation_suffix):
        src_tensor = all_nodes[type_masks[src_type_id]]
        dst_tensor = all_nodes[type_masks[dst_type_id]]
        src_offset = torch.where(type_masks[src_type_id])[0]
        dst_offset = torch.where(type_masks[dst_type_id])[0]

        if src_tensor.size(0) == 0 or dst_tensor.size(0) == 0:
            edge_index_dict[("Node", relation_suffix, "Node")] = torch.zeros((2, 0), dtype=torch.long)
            edge_attr_dict[("Node", relation_suffix, "Node")] = torch.zeros((0, 1), dtype=torch.float32)
            return

        src_x, src_y = src_tensor[:, 0], src_tensor[:, 1]
        dst_x, dst_y = dst_tensor[:, 0], dst_tensor[:, 1]

        src, dst, attr = [], [], []
        for i in range(len(src_x)):
            for j in range(len(dst_x)):
                dist = torch.norm(torch.tensor([src_x[i] - dst_x[j], src_y[i] - dst_y[j]]))
                weight = math.exp(-dist.item()) if dst_type_id == 2 else 1.0 / (1.0 + dist.item())
                if weight > normalized_threshold:
                    src.append(src_offset[i].item())
                    dst.append(dst_offset[j].item())
                    attr.append(weight)

        edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(attr, dtype=torch.float32).unsqueeze(1) if attr else torch.zeros((0, 1))
        edge_index_dict[("Node", relation_suffix, "Node")] = edge_index
        edge_attr_dict[("Node", relation_suffix, "Node")] = edge_attr

    add_edges(0, 0, "attk_and_attk")
    add_edges(0, 1, "attk_and_def")
    add_edges(1, 1, "def_and_def")
    add_edges(0, 2, "attk_and_ball")
    add_edges(1, 2, "def_and_ball")
    
    return edge_index_dict, edge_attr_dict


def convert_to_hetero_graph(node_features, edge_index_dict, edge_attr_dict):
    data = HeteroData()
    for node_type, feats in node_features.items():
        data[node_type].x = feats
    for edge_type, edge_index in edge_index_dict.items():
        data[edge_type].edge_index = edge_index
        data[edge_type].edge_attr = edge_attr_dict[edge_type]
    return data

def build_graph_sequence_from_condition(sample):
    condition = sample["condition"]     # [T, F]
    T = condition.shape[0]

    full_graph = HeteroData()
    node_offset = 0

    added_rels = set()

    for t in range(T):
        # print(f"[Frame {t+1}/{T}] node_offset_before={node_offset}")
        node_feats = extract_node_features(condition[t], sample["condition_columns"])
        edge_index_dict, edge_attr_dict = build_edges_based_on_interactions(node_feats)

        node_count = node_feats["Node"].size(0)
        if t == 0:
            full_graph["Node"].x = node_feats["Node"]
        else:
            full_graph["Node"].x = torch.cat(
                [full_graph["Node"].x, node_feats["Node"]], dim=0)

        for rel, eidx in edge_index_dict.items():
            offset_eidx = eidx + node_offset
            # rel is tuple like ("Node","attk_and_def","Node")
            if rel in added_rels:
                full_graph[rel].edge_index = torch.cat(
                    [full_graph[rel].edge_index, offset_eidx], dim=1)
                full_graph[rel].edge_attr = torch.cat(
                    [full_graph[rel].edge_attr, edge_attr_dict[rel]], dim=0)
            else:
                full_graph[rel].edge_index = offset_eidx
                full_graph[rel].edge_attr = edge_attr_dict[rel]
                added_rels.add(rel)

        if t > 0:
            prev_offset = node_offset - node_count
            tem_edges = torch.stack([
                torch.arange(node_count, device=node_feats["Node"].device) + prev_offset,
                torch.arange(node_count, device=node_feats["Node"].device) + node_offset
            ])
            rel = ("Node", "temporal", "Node")
            ones_attr = torch.ones((node_count,1), device=tem_edges.device)
            if rel in added_rels:
                full_graph[rel].edge_index = torch.cat(
                    [full_graph[rel].edge_index, tem_edges], dim=1)
                full_graph[rel].edge_attr = torch.cat(
                    [full_graph[rel].edge_attr, ones_attr], dim=0)
            else:
                full_graph[rel].edge_index = tem_edges
                full_graph[rel].edge_attr = ones_attr
                added_rels.add(rel)

        node_offset += node_count

    return full_graph

