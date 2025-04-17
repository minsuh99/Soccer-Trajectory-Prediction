import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, HeteroConv
from torch_geometric.data import HeteroData
from torch_geometric.utils import softmax

class EdgeWeightedGATv2Conv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=2, concat=False, negative_slope=0.2):
        super().__init__(aggr='add', node_dim=0)
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.heads = heads
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index, edge_attr):
        x = self.lin(x).view(-1, self.heads, self.out_channels)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        h = x_i + x_j
        score = F.leaky_relu((h * self.att).sum(dim=-1), negative_slope=self.negative_slope)
        score = score * edge_attr
        alpha = softmax(score, index, ptr, size_i)
        return x_j * alpha.unsqueeze(-1)

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, x):
        scores = x @ self.query
        weights = torch.softmax(scores, dim=0).unsqueeze(-1)
        return (weights * x).sum(dim=0)

class InteractionGraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=128, heads=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.pool = AttentionPooling(hidden_dim)

        etypes = [
            ('Node','attk_and_attk','Node'),
            ('Node','attk_and_def','Node'),
            ('Node','def_and_def','Node'),
            ('Node','attk_and_ball','Node'),
            ('Node','def_and_ball','Node'),
            ('Node','temporal','Node')
        ]
        conv1 = {etype: EdgeWeightedGATv2Conv(in_dim, hidden_dim, heads=heads, concat=False) for etype in etypes}
        conv2 = {etype: EdgeWeightedGATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False) for etype in etypes}
        self.hetero_conv1 = HeteroConv(conv1, aggr='sum')
        self.hetero_conv2 = HeteroConv(conv2, aggr='sum')
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, graph: HeteroData):
        x = graph['Node'].x
        
        x_dict = {'Node': x}
        x_dict = self.hetero_conv1(x_dict, graph.edge_index_dict, edge_attr_dict=graph.edge_attr_dict)
        h = F.relu(x_dict['Node'])
        x = self.norm1(h)
        
        x_dict = {'Node': x}
        x_dict = self.hetero_conv2(x_dict, graph.edge_index_dict, edge_attr_dict=graph.edge_attr_dict)
        h = F.relu(x_dict['Node'])
        x = self.norm2(h)

        graph['Node'].x = x
        
        graph_rep = self.pool(x)
        graph_rep = self.out_proj(graph_rep)
        
        return graph_rep
