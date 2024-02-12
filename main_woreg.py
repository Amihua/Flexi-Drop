# encoding = utf-8

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import AdjSampler, read_saint_data
from model import GCN
from tensorboardX import summary
from tqdm import tqdm
from utils import Radamacher_Regularization_p_inf_q_1

WANDB = False


def get_parser():
    parser = argparse.ArgumentParser(description = "Train")
    parser.add_argument("--num-layers", type = int, default = 2)
    parser.add_argument("--num-neighbors", type = list, default = [2, 2])
    parser.add_argument("--hidden-size", type = int, default = 256)
    parser.add_argument("--batch-size", type = int, default = 2048)
    parser.add_argument("--dropout", type = float, default = 0.5)
    parser.add_argument("--lr", type = float, default = 0.001)
    parser.add_argument("--weight-decay", type = float, default = 1e-5)
    parser.add_argument("--epochs", type = int, default = 100)
    parser.add_argument("--runs", type = int, default = 10)
    parser.add_argument("--dataset", type = int, default = 10)
    args = parser.parse_args()
    return args


args = get_parser()
if WANDB:
    import wandb
    wandb.init(
        project = 'Graph',
        config = vars(args),
        save_code = True,
        group = 'GCN'
    )
main_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
data_dir = os.path.join(main_dir, 'dataset', 'arxiv')
data_dir = r"D:\github\graph\dataset\cora"
feats, labels, edge_index, train_mask, val_mask, test_mask = read_saint_data(folder = data_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = GCN(nfeat = feats.shape[1],
            nhid = args.hidden_size,
            nclass = labels.max().item() + 1,
            dropout = args.dropout)

optimizer = torch.optim.Adam(model.parameters(),
                             lr = args.lr, weight_decay = args.weight_decay)

criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(feats, edge_index)  # Perform a single forward pass.
    # import pdb;pdb.set_trace()
    loss = criterion(out[train_mask], labels[train_mask].long())  # Compute the loss solely based on the training nodes.
    extra_loss = 0.5 * Radamacher_Regularization_p_inf_q_1(
        model, feats, labels.max().item() + 1)
    (loss + extra_loss).backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss, extra_loss


@torch.no_grad()
def val():
    model.eval()
    out = model(feats, edge_index)
    pred = out.argmax(dim = 1)  # Use the class with highest probability.
    val_correct = pred[val_mask] == labels[val_mask]  # Check against ground-truth labels.
    val_acc = int(val_correct.sum()) / int(val_mask.sum())  # Derive ratio of correct predictions.
    return val_acc


@torch.no_grad()
def test():
    model.eval()
    out = model(feats, edge_index)
    pred = out.argmax(dim = 1)  # Use the class with highest probability.
    test_correct = pred[test_mask] == labels[test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(test_mask.sum())  # Derive ratio of correct predictions.
    return test_acc


for epoch in range(1, 101):
    loss, extra_loss = train()
    val_acc = val()
    test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Reg: {extra_loss:.4f}, val_acc:{val_acc:.4f}, test_acc: {test_acc:.4f}')
    if WANDB:
        res = {}
        for item in ['epoch', 'loss', 'extra_loss', 'val_acc', 'test_acc']:
            res[item] = eval(item)
        wandb.log(res)
