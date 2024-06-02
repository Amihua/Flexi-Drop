import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

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
    def __init__(self, nfeat, nhid, nclass, dropout, use_vardrop = True):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nclass)
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


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, use_vardrop = True):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(nfeat, nhid)
        self.conv2 = GATConv(nhid, int(nclass))
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