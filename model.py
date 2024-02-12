import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn
from torch_geometric.nn import APPNP, GATConv, GCNConv, GINConv, SAGEConv
from torch_geometric.nn import Sequential as GSequential


class VariationalDropout(nn.Module): 
    """
    Class for Dropout layer
    Args:
        initial_rates (torch.cuda.tensor): initial points for retain probabilities for
                                            Bernoulli dropout layer
    mode (str): 'deterministic' or 'stochastic'
    """

    def __init__(self, initial_rates, mode):
        super(VariationalDropout, self).__init__()

        self.mode = mode
        #self.probs = torch.nn.Parameter(initial_rates).cuda()
        self.p = torch.nn.Parameter(initial_rates)

    def forward(self, input):

        if self.mode == 'stochastic':
            mask = torch.bernoulli(torch.clamp(self.p.data, 0, 1)).view(1, input.shape[1])

        elif self.mode == 'deterministic':
            mask = torch.clamp(self.p, 0, 1).view(1, input.shape[1])

        else:
            raise Exception("Check mode: stochastic or deterministic only")

        return input * mask

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, drop_p, use_vardrop = True):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nclass)
        if use_vardrop:
            self.dropout = VariationalDropout(torch.Tensor([self.dropout] * nhid), 'deterministic')
        else:
            self.dropout = nn.Dropout(p=drop_p)
        self.use_vardrop = use_vardrop
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

class APPNP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, drop_p, use_vardrop = True):
        super().__init__()
        self.lin1 = nn.Linear(nfeat, nhid)
        self.lin2 = nn.Linear(nhid, nclass)
        self.dropout = dropout
        self.appnp = torch_geometric.nn.APPNP(K=2, alpha=0.1)
        if use_vardrop:
            self.dropout_layer = VariationalDropout(torch.Tensor([self.dropout] * nhid), 'deterministic')
        else:
            self.dropout_layer = nn.Dropout(p=drop_p)
        self.use_vardrop = use_vardrop
    def forward(self, x, edge_index):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        x = self.lin2(x)
        x = self.appnp(x, edge_index)
        return x



class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, drop_p, use_vardrop = True):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(nfeat, nhid)
        self.conv2 = GATConv(nhid, nclass)
        if use_vardrop:
            self.dropout = VariationalDropout(torch.Tensor([self.dropout] * nhid), 'deterministic')
        else:
            self.dropout = nn.Dropout(p=drop_p)
        self.use_vardrop = use_vardrop
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

class SAGE(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, use_vardrop = True):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(nfeat, nhid)
        self.conv2 = SAGEConv(nhid, nclass)
        if use_vardrop:
            self.dropout = VariationalDropout(torch.Tensor([self.dropout] * nhid), 'deterministic')
        else:
            self.dropout = nn.Dropout()
        self.use_vardrop = use_vardrop
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

class GIN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, drop_p, num_layers, use_vardrop = True):
        super().__init__()
        self.dropout = dropout
       
        if use_vardrop:
            self.dropout = VariationalDropout(torch.Tensor([self.dropout] * nhid), 'deterministic')
        else:
            self.dropout = nn.Dropout(p=drop_p)

        self.use_vardrop = use_vardrop

        layers = []
        for i in range(num_layers):
            input_dim = nfeat if i == 0 else nhid
            layers.append((GINConv(nn.Sequential(
                nn.Linear(input_dim, nhid),
                # nn.ReLU(),
                # nn.Linear(nhid, nhid)
            )), 'x, edge_index -> x'))
            layers.append(nn.ReLU())
            layers.append(self.dropout)

        self.layers = GSequential('x, edge_index', layers)
        self.lin = nn.Linear(nhid, nclass)

    def forward(self, x, edge_index):
        x = self.layers(x, edge_index)
        x = self.lin(x)
        
        return x
    
# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout):
#         super(GCN, self).__init__()
#
#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nhid, nclass)
#         self.dropout = dropout
#         self.nhid = nhid
#         self.VarDropout = VariationalDropout(torch.Tensor([self.dropout] * self.nhid), 'deterministic')
#
#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj))
#         x = self.VarDropout(x)
#         x = self.gc2(x, adj)
#         return F.log_softmax(x, dim=1)
