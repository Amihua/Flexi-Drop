import numpy as np
import scipy.sparse as sp
import torch
from model import GCNConv, VariationalDropout
from numpy import shape


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path=r"D:\github\graph\Radegcn_inf1\data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def Radamacher_Regularization_p_inf_q_1(net, X_batch, nclass):
    """
    Calculates p_inf_q_1 Radamacher Regularization for the model,
    discussed in the appendix of the article https://openreview.net/pdf?id=S1uxsye0Z
    Args:
        net: neural network, the last layer should be fC,
                                    with output (number_elements, number_of_classes)
        X_batch (torch.Tensor): Sample matrix, size (batch_size, features)

    Return:
        loss (torch.tensor): Radamacher Regularization in the form p_inf_q_ 1
    """
    n, d = X_batch.shape[0], X_batch.shape[1]

    #k = net[-1].weight.shape[0]
    k = nclass

    loss = torch.max(torch.abs(X_batch)) * k * np.sqrt(np.log(d) / n)

    for layer in net.children():
        # Take retain probs from VariationalDropout class
        if isinstance(layer, VariationalDropout) or isinstance(layer, torch.nn.Dropout):
            if torch.is_tensor(layer.p):
                retain_probability = torch.clamp(layer.p, 0, 1)
            else:
                retain_probability = torch.tensor(layer.p).to(X_batch.device)
            loss *= torch.sum(torch.abs(retain_probability))
            print('shape', retain_probability.max())
        # Take weight from FC layers
        elif isinstance(layer, GCNConv):
            loss *= 2 * torch.max(torch.abs(layer.lin.weight))

            k_new, k_old = layer.lin.weight.shape

            loss *= np.sqrt(k_new + k_old) / k_new

    return loss


def Radamacher_Regularization_dc(net, X_batch, nclass):

    n, d = X_batch.shape[0], X_batch.shape[1]

    k = nclass

    loss = torch.max(torch.abs(X_batch)) * k * np.sqrt(2 * np.log(2*d) / n)
    for layer in net.children():
        # Take retain probs from VariationalDropout class
        if isinstance(layer, VariationalDropout):
            retain_probability = torch.clamp(layer.p, 0, 1)
            loss *= torch.norm(retain_probability, p = 2, dim = -1)
            # print('retain rate', retain_probability.max())
        elif isinstance(layer, torch.nn.Dropout):
            retain_probability = torch.tensor(layer.p).to(X_batch.device)
            loss *= torch.norm(retain_probability, p = 2, dim = -1)
            # print('retain rate', retain_probability.max())
        elif isinstance(layer, GCNConv):
            loss *= 2 * torch.max(torch.norm(layer.lin.weight, p = 2, dim = 0))
            k_new, k_old = layer.lin.weight.shape
            loss *= np.sqrt(k_new + k_old) / k_new

    return loss
