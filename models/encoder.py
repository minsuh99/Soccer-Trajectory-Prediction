import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv

class GraphAutoencoder(nn.Module):
    def __init__(self, in_dim_dict, hidden_dim, temporal_hidden_dim, out_dim):
        super().__init__()

        self.node_proj = nn.ModuleDict({
            k: nn.Linear(in_dim_dict[k], hidden_dim) for k in in_dim_dict
        })

        self.spatial_encoding = HeteroConv({
            ('Attk', 'interaction', 'Attk'): GCNConv(hidden_dim, hidden_dim, add_self_loops=False),
            ('Attk', 'interaction', 'Def'): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            ('Def', 'interaction', 'Def'): GCNConv(hidden_dim, hidden_dim, add_self_loops=False),
            ('Attk', 'interaction', 'Ball'): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            ('Def', 'interaction', 'Ball'): SAGEConv((hidden_dim, hidden_dim), hidden_dim)
        }, aggr='sum')

        self.temporal_encoding = nn.GRU(
            input_size=hidden_dim, hidden_size=temporal_hidden_dim, batch_first=True
        )

        self.h_proj = nn.Linear(temporal_hidden_dim * len(in_dim_dict), out_dim)

        self.h_to_z = nn.ModuleDict({
            k: nn.Linear(out_dim, temporal_hidden_dim) for k in in_dim_dict
        })

        self.node_decoder = nn.ModuleDict({
            k: nn.ModuleDict({
                "continuous": nn.Linear(temporal_hidden_dim, 5),     # x, y, vx, vy, dist
                "position": nn.Linear(temporal_hidden_dim, 24),      # 포지션 logits
                "starter": nn.Linear(temporal_hidden_dim, 1)         # 선발 여부 (binary)
            }) for k in in_dim_dict
        })

    def encode_frame(self, data):
        x_dict = {}
        for node_type in data.x_dict:
            x = data[node_type].x
            x_proj = self.node_proj[node_type](x)
            x_dict[node_type] = x_proj
        z_dict = self.spatial_encoding(x_dict, data.edge_index_dict)
        return z_dict

    def forward(self, graph_seq):
        z_seq_dict = {k: [] for k in graph_seq[0].x_dict.keys()}

        for t in range(len(graph_seq)):
            z_t = self.encode_frame(graph_seq[t])
            for k in z_seq_dict:
                z_seq_dict[k].append(z_t[k])

        pooled = []
        for k in z_seq_dict:
            z_seq = torch.stack(z_seq_dict[k], dim=1)
            _, h_n = self.temporal_encoding(z_seq)
            h_last = h_n.squeeze(0).mean(dim=0)
            pooled.append(h_last)

        H = self.h_proj(torch.cat(pooled, dim=-1))
        return H

    def decode_from_H(self, H, data):
        z_final_hat = {
            k: self.h_to_z[k](H).expand(data[k].num_nodes, -1)
            for k in data.x_dict
        }

        continuous_loss = 0.0
        categorical_loss = 0.0

        for k in z_final_hat:
            z = z_final_hat[k]
            x_gt = data[k].x.to(H.device)

            x_hat_cont = self.node_decoder[k]["continuous"](z)
            x_hat_pos = self.node_decoder[k]["position"](z)    # [N, 24]
            x_hat_starter = self.node_decoder[k]["starter"](z) # [N, 1]

            # --- Continuous loss ---
            continuous_loss += F.mse_loss(x_hat_cont, x_gt[:, :5])

            # --- Categorical (multi-class: position) ---
            position_label = x_gt[:, 5].long()  # (1~23)
            categorical_loss += F.cross_entropy(x_hat_pos, position_label)

            # --- Categorical (binary: starter 여부) ---
            starter_label = x_gt[:, 6]
            categorical_loss += F.binary_cross_entropy_with_logits(
                x_hat_starter.squeeze(-1), starter_label.float()
            )

        # --- Edge loss 그대로 유지 ---
        edge_loss = 0.0
        for (src, _, dst), edge_index in data.edge_index_dict.items():
            z_src, z_dst = z_final_hat[src], z_final_hat[dst]
            u, v = edge_index
            pos_score = (z_src[u] * z_dst[v]).sum(dim=-1)
            pos_label = torch.ones_like(pos_score)

            u_neg = torch.randint(0, z_src.size(0), (u.size(0),), device=u.device)
            v_neg = torch.randint(0, z_dst.size(0), (v.size(0),), device=v.device)
            neg_score = (z_src[u_neg] * z_dst[v_neg]).sum(dim=-1)
            neg_label = torch.zeros_like(neg_score)

            score = torch.cat([pos_score, neg_score], dim=0)
            label = torch.cat([pos_label, neg_label], dim=0)
            edge_loss += F.binary_cross_entropy_with_logits(score, label)

        return continuous_loss + categorical_loss + edge_loss



