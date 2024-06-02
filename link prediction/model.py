import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
#from .... import function as fn
# import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch import GraphConv,GATConv,SAGEConv

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

class GCN(nn.Module): #ok
    def __init__(self, in_dim, hid_dim, out_dim, dropout, use_vardrop):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GraphConv(in_dim, hid_dim)
        self.conv2 = GraphConv(hid_dim, out_dim)
        if use_vardrop:
            self.dropout = VariationalDropout(torch.Tensor([self.dropout] * hid_dim), 'deterministic')
        else:
            self.dropout = nn.Dropout()
        self.use_vardrop = use_vardrop

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.conv1, gain=gain)
        nn.init.xavier_normal_(self.conv2, gain=gain)
        nn.init.xavier_normal_(self.dropout, gain=gain)

    def forward(self, blocks, h):
        x = F.relu(self.conv1(blocks[0], h))
        x = self.dropout(x)
        x = self.conv2(blocks[1], x)
        return x


class GAT(nn.Module): #ok
    def __init__(self, in_dim, hid_dim, out_dim, dropout, use_vardrop):
        super().__init__()
        self.dropout = dropout
        self.num_heads = 1
        self.conv1 = GATConv(in_dim, hid_dim,self.num_heads)  # ,dropout=0.6
        self.conv2 = GATConv(hid_dim , out_dim, self.num_heads)
        if use_vardrop:
            self.dropout = VariationalDropout(torch.Tensor([self.dropout] * hid_dim), 'deterministic')
        else:
            self.dropout = nn.Dropout()
        self.use_vardrop = use_vardrop
    def forward(self, blocks, h):
        x = F.elu(self.conv1(blocks[0], h))
        x = self.dropout(x)
        x = self.conv2(blocks[1], x)
        return x.squeeze()

class Graphsage(nn.Module): #ok
    def __init__(self, in_dim, hid_dim, out_dim, dropout, use_vardrop):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(in_dim,hid_dim, 'mean')
        self.conv2 = SAGEConv(hid_dim,out_dim, 'mean')
        if use_vardrop:
            self.dropout = VariationalDropout(torch.Tensor([self.dropout] * hid_dim), 'deterministic')
        else:
            self.dropout = nn.Dropout(0.5)
        self.use_vardrop = use_vardrop
    def forward(self, blocks, h):
        x = F.relu(self.conv1(blocks[0], h))
        x = self.dropout(x)
        x = self.conv2(blocks[1], x)
        return x