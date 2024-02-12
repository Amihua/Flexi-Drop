import copy
import json
import os.path as osp
import random
import time

import numpy as np
import scipy.sparse as sp
import torch
from cogdl.data import Dataset, Graph
from sklearn.preprocessing import StandardScaler


def add_random_edges(edge_index, num_nodes, p_edge):
    num_edges = edge_index.size(1)
    additional_edges = int(num_edges * p_edge)
    
    new_edges = set()
    while len(new_edges) < additional_edges:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u != v and (u, v) not in new_edges and (v, u) not in new_edges:
            new_edges.add((u, v))

    additional_edge_index = torch.tensor(list(new_edges)).t().contiguous()
    edge_index = torch.cat([edge_index, additional_edge_index], dim=1)
    return edge_index


class PseudoRanger(torch.utils.data.Dataset):
    def __init__(self, num):
        self.indices = torch.arange(num)
        self.num = num

    def __getitem__(self, item):
        return self.indices[item]

    def __len__(self):
        return self.num

    def shuffle(self):
        rand = torch.randperm(self.num)
        self.indices = self.indices[rand]


class AdjSampler(torch.utils.data.DataLoader):
    def __init__(self, graph, sizes = [2, 2], training = True, *args, **kwargs):

        self.graph = copy.deepcopy(graph)
        self.sizes = sizes
        self.degree = graph.degrees()
        self.diag = self._sparse_diagonal_value(graph)
        self.training = training
        if training:
            idx = torch.where(graph['train_mask'])[0]
        else:
            idx = torch.arange(0, graph.x.shape[0])
        self.dataset = PseudoRanger(idx.shape[0])

        kwargs["collate_fn"] = self.collate_fn
        super(AdjSampler, self).__init__(self.dataset, *args, **kwargs)

    def shuffle(self):
        self.dataset.shuffle()

    def _sparse_diagonal_value(self, adj):
        row, col = adj.edge_index
        value = adj.edge_weight
        return value[row == col]

    def _construct_propagation_matrix(self, sample_adj, sample_id, num_neighbors):
        row, col = sample_adj.edge_index
        value = sample_adj.edge_weight
        """add self connection"""
        num_row = row.max() + 1
        row = torch.cat([torch.arange(0, num_row).long(), row], dim = 0)
        col = torch.cat([torch.arange(0, num_row).long(), col], dim = 0)
        value = torch.cat([self.diag[sample_id[:num_row]], value], dim = 0)

        value = value * self.degree[sample_id[row]] / num_neighbors
        new_graph = Graph()
        new_graph.edge_index = torch.stack([row, col])
        new_graph.edge_weight = value
        return new_graph

    def collate_fn(self, idx):
        if self.training:
            sample_id = torch.tensor(idx)
            sample_adjs, sample_ids = [], [sample_id]
            full_adjs, full_ids = [], []

            for size in self.sizes:
                full_id, full_adj = self.graph.sample_adj(sample_id, -1)
                sample_id, sample_adj = self.graph.sample_adj(sample_id, size, replace = False)

                sample_adj = self._construct_propagation_matrix(sample_adj, sample_id, size)

                sample_adjs = [sample_adj] + sample_adjs
                sample_ids = [sample_id] + sample_ids
                full_adjs = [full_adj] + full_adjs
                full_ids = [full_id] + full_ids

            return torch.tensor(idx), (sample_ids, sample_adjs), (full_ids, full_adjs)
        else:
            # only return full adj in Evalution phase
            sample_id = torch.tensor(idx)
            full_id, full_adj = self.graph.sample_adj(sample_id, -1)
            return sample_id, full_id,



def index_to_mask(index, size):
    mask = torch.full((size,), False, dtype=torch.bool)
    mask[index] = True
    return mask

def read_saint_data(folder, random_edge=False, p_edge=0.1):
    names = ["adj_full.npz", "adj_train.npz", "class_map.json", "feats.npy", "role.json"]
    names = [osp.join(folder, name) for name in names]
    adj_full = sp.load_npz(names[0])
    adj_train = sp.load_npz(names[1])
    class_map = json.load(open(names[2]))
    feats = np.load(names[3])
    role = json.load(open(names[4]))

    train_mask = index_to_mask(role["tr"], size=feats.shape[0])
    val_mask = index_to_mask(role["va"], size=feats.shape[0])
    test_mask = index_to_mask(role["te"], size=feats.shape[0])

    feats = torch.from_numpy(feats).float()
    item = class_map["0"]
    if isinstance(item, list):
        labels = np.zeros((feats.shape[0], len(item)), dtype=float)
        for key, val in class_map.items():
            labels[int(key)] = np.array(val)
    else:
        labels = np.zeros(feats.shape[0], dtype=np.int64)
        for key, val in class_map.items():
            labels[int(key)] = val

    labels = torch.from_numpy(labels)

    def get_adj(adj):
        row, col = adj.nonzero()
        data = adj.data
        row = torch.tensor(row, dtype=torch.long)
        col = torch.tensor(col, dtype=torch.long)
        edge_index = torch.stack([row, col], dim=0)
        edge_attr = torch.tensor(data, dtype=torch.float)
        return edge_index, edge_attr

    edge_index_full, edge_attr_full = get_adj(adj_full)
    # edge_index_train, edge_attr_train = get_adj(adj_train)

    # data = Graph(
    #     x=feats,
    #     y=labels,
    #     edge_index=edge_index_full,
    #     edge_attr=edge_attr_full,
    #     edge_index_train=edge_index_train,
    #     edge_attr_train=edge_attr_train,
    #     train_mask=train_mask,
    #     val_mask=val_mask,
    #     test_mask=test_mask,
    # )
    train_mask = torch.where(train_mask.bool())[0]
    val_mask = torch.where(val_mask.bool())[0]
    test_mask = torch.where(test_mask.bool())[0]
    if random_edge ==True:
        num_nodes = feats.shape[0]
        edge_index_full = add_random_edges(edge_index_full, num_nodes, p_edge)
    return feats, labels, edge_index_full, train_mask, val_mask, test_mask

if __name__ == '__main__':
    import os
    main_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    data_dir = os.path.join(main_dir, 'dataset', 'cora')
    data = read_saint_data(data_dir)
    pass
