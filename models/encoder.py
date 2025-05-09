import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv
from torch_geometric.data import HeteroData
from torch_geometric.utils import softmax


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_dim) * 0.02)

    def forward(self, x, batch):
        # x: (N, C)
        scores  = (x * self.query).sum(-1)    # (N,)
        weights = softmax(scores, batch)      # (N,)
        out     = weights.unsqueeze(-1) * x   # (N, C)

        B = int(batch.max().item()) + 1
        C = x.size(-1)

        graph_rep = out.new_zeros((B, C))
        graph_rep.index_add_(0, batch, out)

        return graph_rep

class InteractionGraphEncoder(nn.Module):
    """
    A lightweight encoder using built-in GATv2Conv per relation (no custom message-passing).
    """
    def __init__(self, in_dim, hidden_dim=128, out_dim=128, heads=2):
        super().__init__()
        # normalization and pooling
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.pool  = AttentionPooling(hidden_dim)

        # define edge types
        edge_types = [
            ('Node', 'attk_and_attk', 'Node'),
            ('Node', 'attk_and_def', 'Node'),
            ('Node', 'def_and_def', 'Node'),
            ('Node', 'attk_and_ball', 'Node'),
            ('Node', 'def_and_ball', 'Node'),
            ('Node', 'temporal', 'Node'),
        ]
        # 1st layer convs
        conv1 = {
            rel: GATv2Conv(in_dim, hidden_dim, heads=heads, concat=False)
            for rel in edge_types
        }
        # 2nd layer convs
        conv2 = {
            rel: GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False)
            for rel in edge_types
        }

        self.het1 = HeteroConv(conv1, aggr='sum')
        self.het2 = HeteroConv(conv2, aggr='sum')
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, graph: HeteroData):
        x = graph['Node'].x  # (N, F)

        # first heterogeneous attention
        x = self.het1({'Node': x}, graph.edge_index_dict)
        x = x['Node']
        # x = F.relu(x)
        x = F.softsign(x)
        x = self.norm1(x)

        # second heterogeneous attention
        x = self.het2({'Node': x}, graph.edge_index_dict)
        x = x['Node']
        # x = F.relu(x)
        x = F.softsign(x)
        x = self.norm2(x)

        # pooling to graph-level
        batch = graph['Node'].batch
        graph_rep = self.pool(x, batch)
        return self.proj(graph_rep)  # (B, out_dim)
