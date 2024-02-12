# encoding = utf-8
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn
from dataloader import AdjSampler, read_saint_data
from model import APPNP, GAT, GCN, GIN
from tensorboardX import summary
from tqdm import tqdm
from utils import (Radamacher_Regularization_dc,
                   Radamacher_Regularization_p_inf_q_1)

WANDB = True

def calculate_dirichlet_energy(node_features, edge_index):
    # edge_index: [2, E] where E is the number of edges
    # node_features: [N, F] where N is the number of nodes and F is the number of features

    # Retrieve node feature vectors for both the source and the target of each edge
    node_features_source = node_features[edge_index[0], :]
    node_features_target = node_features[edge_index[1], :]

    # Compute the differences between node features for each edge
    differences = node_features_source - node_features_target

    # Compute the squared L2 norm of the differences
    energy = differences.norm(2, dim=1).pow(2).sum()

    # Normalize by the number of nodes
    return energy / node_features.size(0)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)    
    
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.autograd.set_detect_anomaly(True)




def str2bool(x):
    if type(x) is bool:
        return x
    elif type(x) is str:
        return True if  x.lower() in ('true') else False
    elif type(x) is int:
        return True if x >= 1 else 0

def get_parser():
    parser = argparse.ArgumentParser(description = "Train")
    parser.add_argument("--num-layers", type = int, default = 2)
    parser.add_argument("--num-neighbors", type = list, default = [2, 2])
    parser.add_argument("--hidden-size", type = int, default = 128)
    parser.add_argument("--batch-size", type = int, default = 500)
    parser.add_argument("--dropout", type = float, default = 1)
    parser.add_argument("--lr", type = float, default = 0.001)
    parser.add_argument("--weight-decay", type = float, default = 0)
    parser.add_argument("--epochs", type = int, default = 200)
    parser.add_argument("--runs", type = int, default = 10)
    parser.add_argument("--reg-weight", type = float, default = 0.001) 
    parser.add_argument("--use-vardrop", type = str2bool, default = False)
    parser.add_argument("--dropp", type = float, default = 0.6)
    parser.add_argument("--Lambda", type = float, default = 1)
    parser.add_argument("--dataset", type=str, default= 'pubmed')
    parser.add_argument("--pEdge", type = float, default = 0)
    args = parser.parse_args()
    return args

args = get_parser()
'''
    if WANDB:
    import wandb
    wandb.init(
        project = 'Graph',
        config = vars(args),
        save_code = True,
        group = 'DropConnect' if args.use_vardrop else 'GCN',
    )
'''


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
main_dir = os.path.abspath(os.path.join(os.getcwd(), "../2024KDD/"))
data_dir = os.path.join(main_dir, 'dataset', args.dataset)


feats, labels, edge_index, train_mask, val_mask, test_mask = read_saint_data(folder = data_dir, random_edge=False, p_edge=args.pEdge)
feats, labels, edge_index, train_mask, val_mask, test_mask = feats.to(device), labels.to(device), edge_index.to(device), train_mask.to(device), val_mask.to(device), test_mask.to(device)

# model = GCN(nfeat = feats.shape[1],
#             nhid = args.hidden_size,
#             nclass = labels.max().item() + 1,
#             dropout = args.dropout,
#             drop_p=args.dropp,
#             use_vardrop = args.use_vardrop).to(device)

# model = GAT(nfeat = feats.shape[1],
#             nhid = args.hidden_size,
#             nclass = labels.max().item() + 1,
#             dropout = args.dropout,
#             drop_p=args.dropp,
#             use_vardrop = args.use_vardrop).to(device)


layer = 2

model = GIN(nfeat = feats.shape[1],
            nhid = args.hidden_size,
            nclass = labels.max().item() + 1,
            dropout = args.dropout,
            drop_p = args.dropp,
            use_vardrop = args.use_vardrop,
            num_layers = layer).to(device)

optimizer = torch.optim.Adam(model.parameters(),
                            lr = args.lr, weight_decay = args.weight_decay)

criterion = torch.nn.CrossEntropyLoss()

# CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 1000, eta_min=1e-5)
def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(feats, edge_index)  # Perform a single forward pass.
    # import pdb;pdb.set_trace()
    de = calculate_dirichlet_energy(out, edge_index)
    loss = criterion(out[train_mask], labels[train_mask].long())  # Compute the loss solely based on the training nodes.
    extra_loss = (args.reg_weight) * Radamacher_Regularization_dc(
        model, feats, labels.max().item() + 1)
    (loss + args.reg_weight * extra_loss).backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.

    pred = out.argmax(dim = 1)
    train_correct = pred[train_mask] == labels[train_mask]  # Check against ground-truth labels.
    train_acc = int(train_correct.sum()) / train_mask.shape[0]  # Derive ratio of correct predictions.
    # CosineLR.step()
    return loss, extra_loss, train_acc, de


@torch.no_grad()
def val():
    model.eval()
    out = model(feats, edge_index)
    pred = out.argmax(dim = 1)  # Use the class with highest probability.
    val_correct = pred[val_mask] == labels[val_mask]  # Check against ground-truth labels.
    val_acc = int(val_correct.sum()) / val_mask.shape[0]  # Derive ratio of correct predictions.
    return val_acc


@torch.no_grad()
def test():
    model.eval()
    out = model(feats, edge_index)
    pred = out.argmax(dim = 1)  # Use the class with highest probability.
    test_correct = pred[test_mask] == labels[test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / test_mask.shape[0]   # Derive ratio of correct predictions.
    return test_acc


test_accs = []
des = []
for epoch in range(1, args.epochs + 1):
    loss, extra_loss, train_acc, de = train()
    val_acc = val()
    test_acc = test()
    des.append(de)
    #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Reg: {extra_loss:.4f}, train_acc:{train_acc:.4f}, val_acc:{val_acc:.4f}, test_acc: {test_acc:.4f}, dirichlet_energy_value: {de:.4f}')
    test_accs.append(test_acc)
    '''
    if WANDB:
        res = {}
        for item in ['epoch', 'loss', 'extra_loss', 'train_acc', 'val_acc', 'test_acc']:
            res[item] = eval(item)
        wandb.log(res)
    '''
# print(torch.initial_seed(), "best test:",max(test_accs),"de:", des[np.argmax(np.array(test_accs))])
print(layer, "best test:",max(test_accs),"de:", des[np.argmax(np.array(test_accs))])
#print(f'best test: {max(test_accs):.4f}')