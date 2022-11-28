from sqlite3 import apilevel
from telnetlib import EXOPL
from tkinter.messagebox import NO
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
# from loader import BioDataset
# from dataloader import DataLoaderFinetune
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
import numpy as np
        


class GATConv(MessagePassing):
    def __init__(self, emb_dim, p_or_m, device, heads=2, negative_slope=0.2, aggr = "add", input_layer = False):
        super(GATConv, self).__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, emb_dim))

        self.device = device
        self.plus_or_minus = p_or_m

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.node_embeddings = None

        self.input_layer = input_layer
        if self.input_layer:
            self.input_node_embeddings = torch.nn.Embedding(2, emb_dim)
            torch.nn.init.xavier_uniform_(self.input_node_embeddings.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        if self.input_layer:
            x = self.input_node_embeddings(x.to(torch.int64).view(-1,))

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        
        out = self.propagate(self.aggr, edge_index, x=x)
        self.node_embeddings = out
        return out

    def message(self, edge_index, x_i, x_j):   
        alpha = (x_j * self.att).sum(dim=-1)


        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        if self.plus_or_minus == 'minus':
            out = x_i - x_j * alpha.view(-1, self.heads, 1)
            return out 

        elif self.plus_or_minus == 'plus':
            out = x_i + x_j * alpha.view(-1, self.heads, 1)
            return out

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out


class GNN(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gat, graphsage, gcn
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536

    Output:
        node representations

    """
    def __init__(self, p_or_m, device, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ###List of message-passing GNN convs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if layer == 0:
                input_layer = True
            else:
                input_layer = False

            if gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim, p_or_m, device, input_layer = input_layer))

    def forward(self, x, edge_index):
        h_list = [x]

        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index)
            if layer == self.num_layer - 1:
                #remove relu from the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list[1:], dim = 0), dim = 0)[0]

        return node_representation


if __name__ == "__main__":
    pass