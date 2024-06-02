# encoding = utf-8

import torch
import argparse
from dataloader import read_saint_data
from model import GAT
import os
from utils import  Radamacher_Regularization_dc

WANDB = True


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
    parser.add_argument("--lr", type = float, default = 0.01)
    parser.add_argument("--weight-decay", type = float, default = 0)
    parser.add_argument("--epochs", type = int, default = 50)
    parser.add_argument("--runs", type = int, default = 10)
    parser.add_argument("--reg-weight", type = int, default = 0.001)
    parser.add_argument("--use-vardrop", type = str2bool, default = True)
    parser.add_argument("--dataset", type=str, default= 'cora')
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


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
main_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
data_dir = os.path.join(main_dir, 'dataset', args.dataset)
data_dir = '/home/anxy/zzh/2024KDD/dataset/cora'
feats, labels, edge_index, train_mask, val_mask, test_mask = read_saint_data(folder = data_dir)
feats, labels, edge_index, train_mask, val_mask, test_mask = feats.to(device), labels.to(device), edge_index.to(device), train_mask.to(device), val_mask.to(device), test_mask.to(device)


model = GAT(nfeat = feats.shape[1],
            nhid = args.hidden_size,
            nclass = labels.max().item() + 1,
            dropout = args.dropout,
            use_vardrop = args.use_vardrop).to(device)

optimizer = torch.optim.Adam(model.parameters(),
                             lr = args.lr, weight_decay = args.weight_decay)

criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(feats, edge_index)  # Perform a single forward pass.
    # import pdb;pdb.set_trace()
    loss = criterion(out[train_mask], labels[train_mask].long())  # Compute the loss solely based on the training nodes.
    extra_loss = (args.reg_weight) * Radamacher_Regularization_dc(
        model, feats, labels.max().item() + 1)
    (loss + extra_loss).backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.

    pred = out.argmax(dim = 1)
    train_correct = pred[train_mask] == labels[train_mask]  # Check against ground-truth labels.
    train_acc = int(train_correct.sum()) / train_mask.shape[0]  # Derive ratio of correct predictions.

    return loss, extra_loss, train_acc


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


for epoch in range(1, args.epochs + 1):
    loss, extra_loss, train_acc = train()
    val_acc = val()
    test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Reg: {extra_loss:.4f}, train_acc:{train_acc:.4f}, val_acc:{val_acc:.4f}, test_acc: {test_acc:.4f}')
    '''
    if WANDB:
        res = {}
        for item in ['epoch', 'loss', 'extra_loss', 'train_acc', 'val_acc', 'test_acc']:
            res[item] = eval(item)
        wandb.log(res)
    '''