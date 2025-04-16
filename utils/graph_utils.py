import pandas as pd
import torch
from torch_geometric.data import HeteroData
import os
from tqdm import tqdm

def frame_tensor_to_df(frame_tensor, column_names):
    np_array = frame_tensor.cpu().numpy()
    df = pd.DataFrame([np_array], columns=column_names)
    return df

def get_global_dist_min_max(dataset):
    all_dists = []
    for sample in dataset:
        cond = sample["condition"]  # [T, F]
        cols = sample["condition_columns"]

        for t in range(cond.shape[0]):
            for col in cols:
                if col.endswith("_dist"):
                    val = cond[t, cols.index(col)].item()
                    all_dists.append(val)

    all_dists_tensor = torch.tensor(all_dists)
    return all_dists_tensor.min().item(), all_dists_tensor.max().item()
# Node feature
def extract_node_features_from_condition(condition_tensor, condition_columns, dist_min=0.0, dist_max=6000.0):
    column_index_map = {col: idx for idx, col in enumerate(condition_columns)}
    player_bases = sorted(list(set(col.rsplit("_", 1)[0] for col in condition_columns if "ball" not in col)))
    attk_bases = [base for base in player_bases if "Attk" in base or any(f"{base}_position" in col for col in condition_columns[:22*7])][:11]
    def_bases = [base for base in player_bases if base not in attk_bases][:11]

    node_features = {"Attk": [], "Def": []}

    def normalize_dist(val):
        return (val - dist_min) / (dist_max - dist_min + 1e-6)

    for base in attk_bases:
        feats = []
        for feat in ["x", "y", "vx", "vy", "dist", "position", "starter"]:
            col = f"{base}_{feat}"
            val = condition_tensor[column_index_map[col]] if col in column_index_map else torch.tensor(0.0)
            if feat == "dist":
                val = normalize_dist(val)
            feats.append(val)
        node_features["Attk"].append(torch.stack(feats))

    for base in def_bases:
        feats = []
        for feat in ["x", "y", "vx", "vy", "dist", "position", "starter"]:
            col = f"{base}_{feat}"
            val = condition_tensor[column_index_map[col]] if col in column_index_map else torch.tensor(0.0)
            if feat == "dist":
                val = normalize_dist(val)
            feats.append(val)
        node_features["Def"].append(torch.stack(feats))

    ball_feats = []
    for feat in ["x", "y", "vx", "vy"]:
        col = f"ball_{feat}"
        val = condition_tensor[column_index_map[col]] if col in column_index_map else torch.tensor(0.0)
        ball_feats.append(val)
    node_features["Ball"] = torch.stack(ball_feats).unsqueeze(0)

    node_features["Attk"] = torch.stack(node_features["Attk"]) if node_features["Attk"] else torch.empty((0, 7))
    node_features["Def"] = torch.stack(node_features["Def"]) if node_features["Def"] else torch.empty((0, 7))

    return node_features

# Edge
def build_edges_based_on_interactions(node_features):
    edge_index_dict = {}
    edge_attr_dict = {}
    
    field_x_half = 52.5
    field_y_half = 34.0
    x_scale = 1.0 / field_x_half
    y_scale = 1.0 / field_y_half
    real_threshold = 20  # Distance-based weight (20 m)

    normalized_threshold = (real_threshold**2 * x_scale**2 + real_threshold**2 * y_scale**2)**0.5

    def add_edges(src_type, dst_type, relation):
        src_tensor = node_features[src_type]
        dst_tensor = node_features[dst_type]

        if src_tensor.size(0) == 0 or dst_tensor.size(0) == 0:
            edge_index_dict[(src_type, relation, dst_type)] = torch.zeros((2, 0), dtype=torch.long)
            edge_attr_dict[(src_type, relation, dst_type)] = torch.zeros((0, 1), dtype=torch.float32)
            return

        src_x, src_y = src_tensor[:, 0], src_tensor[:, 1]
        dst_x, dst_y = dst_tensor[:, 0], dst_tensor[:, 1]

        src, dst, attr = [], [], []
        for i in range(len(src_x)):
            for j in range(len(dst_x)):
                dist = torch.norm(torch.tensor([src_x[i] - dst_x[j], src_y[i] - dst_y[j]]))
                if dist < 20:
                    src.append(i)
                    dst.append(j)
                    attr.append(dist.item())

        edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(attr, dtype=torch.float32).unsqueeze(1) if attr else torch.zeros((0, 1))
        edge_index_dict[(src_type, relation, dst_type)] = edge_index
        edge_attr_dict[(src_type, relation, dst_type)] = edge_attr

    add_edges("Attk", "Attk", "interaction")
    add_edges("Attk", "Def", "interaction")
    add_edges("Def", "Def", "interaction")
    add_edges("Attk", "Ball", "interaction")
    add_edges("Def", "Ball", "interaction")

    return edge_index_dict, edge_attr_dict

def convert_to_hetero_graph(node_features, edge_index_dict, edge_attr_dict):
    data = HeteroData()
    for node_type, feats in node_features.items():
        data[node_type].x = feats
    for edge_type, edge_index in edge_index_dict.items():
        data[edge_type].edge_index = edge_index
        data[edge_type].edge_attr = edge_attr_dict[edge_type]
    return data

def build_graph_sequence_from_condition(sample, data_root=None):
    condition = sample["condition"]     # [T, F]
    condition_columns = sample["condition_columns"]

    graph_seq = []
    for t in range(condition.shape[0]):
        node_feats = extract_node_features_from_condition(condition[t], condition_columns)
        edge_index_dict, edge_attr_dict = build_edges_based_on_interactions(node_feats)
        graph = convert_to_hetero_graph(node_feats, edge_index_dict, edge_attr_dict)
        graph_seq.append(graph)

    return graph_seq

def save_condition_graphs_for_dataset(dataset, indices, save_root="graph_cache"):
    os.makedirs(save_root, exist_ok=True)
    cache = {}
    for idx in tqdm(indices, desc="Saving graph sequences"):
        sample = dataset[idx]
        match_id = sample["match_id"]
        segment_id = f"{match_id}_{sample['condition_frames'][0]}"

        if segment_id in cache:
            continue

        graph_seq = build_graph_sequence_from_condition(sample)
        torch.save(graph_seq, os.path.join(save_root, f"{segment_id}.pt"))
        cache[segment_id] = True
