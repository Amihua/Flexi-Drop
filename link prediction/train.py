import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
#from model import GNN
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import argparse
from model import GCN, VariationalDropout,GraphConv,SAGEConv



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
            #print('shape', retain_probability.max())
        # Take weight from FC layers
        elif isinstance(layer, GraphConv):
            loss *= 2 * torch.max(torch.abs(layer.fc))

            k_new, k_old = layer.fc.shape

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
            #print('retain rate', retain_probability.max())
        elif isinstance(layer, torch.nn.Dropout):
            retain_probability = torch.tensor(layer.p).to(X_batch.device)
            loss *= torch.norm(retain_probability, p = 2, dim = -1)
            #print('retain rate', retain_probability.max())
        elif isinstance(layer, GraphConv):
            loss *= 2 * torch.max(torch.norm(layer.weight, p = 2, dim = 0))
            k_new, k_old = layer.weight.shape
            loss *= np.sqrt(k_new + k_old) / k_new

    return loss

def logs(log_file, content):
    with open(os.path.join('/home/anxy/zzh/Dropconnect/link prediction/result', log_file), mode='a') as f:
        f.write(content + '\t\n')

def Loss(block_outputs, pos_graph, neg_graph, neg_num):
    loss = []
    with pos_graph.local_scope():
        pos_graph.ndata['h'] = block_outputs
        pos_graph.apply_edges(fn.u_dot_v('h', 'h', 'score')) 
        pos_score = pos_graph.edata['score']
    with neg_graph.local_scope():
        neg_graph.ndata['h'] = block_outputs
        neg_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
        neg_score = neg_graph.edata['score']

    pos_loss = F.logsigmoid(pos_score).squeeze(dim=1)
    negs_loss = F.logsigmoid(-neg_score).squeeze(dim=1)
    loss = -(pos_loss.mean() + neg_num*negs_loss.mean())
    return loss

def ACC(block_outputs, pos_graph, neg_graph):
    with pos_graph.local_scope():
        pos_graph.ndata['h'] = block_outputs
        pos_graph.apply_edges(fn.u_dot_v('h', 'h', 'score')) 
        pos_score = pos_graph.edata['score']
    with neg_graph.local_scope():
        neg_graph.ndata['h'] = block_outputs
        neg_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
        neg_score = neg_graph.edata['score']

    pos_loss = F.sigmoid(pos_score).squeeze(dim=1)
    pos_loss[pos_loss>0.5] = 1
    pos_loss[pos_loss<0.5] = 0
    negs_loss = F.sigmoid(neg_score).squeeze(dim=1)
    negs_loss[negs_loss>0.5] = 1
    negs_loss[negs_loss<0.5] = 0
    negs_loss = len(negs_loss) - negs_loss.sum().item()
    acc = (pos_loss.sum().item() + negs_loss)/(2*len(pos_loss))
    return acc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = 'cora' # cora citeseer pubmed
graph_path =  '/home/anxy/zzh/Dropconnect/link prediction/'+dataset+'/'+dataset+'.graph'

graph_list, _ = dgl.load_graphs(graph_path)
graph = graph_list[0]
# graph = graph.to(device)
graph = dgl.add_self_loop(graph)
graph.edata['train_mask'][graph.num_edges() - graph.num_nodes():graph.num_edges()] = 0
graph.edata['test_mask'][graph.num_edges() - graph.num_nodes():graph.num_edges()] = 0
graph = graph.to(device)

train_mask = graph.edata['train_mask'] #8687
train_seeds = torch.nonzero(train_mask).squeeze() # torch.Size([2542888])
train_seeds = torch.tensor(train_seeds,dtype=torch.int32).to(device)
fanouts = [10,10] ## 每一层，每个类型的边采样10个邻居
sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
negs_number = 1
dataloader = dgl.dataloading.EdgeDataLoader(
graph, train_seeds, sampler,
negative_sampler=dgl.dataloading.negative_sampler.Uniform(negs_number),
batch_size=1024,
shuffle=True,
drop_last=False,
num_workers=0)

test_mask = graph.edata['test_mask'] #2171
test_seeds = torch.nonzero(test_mask).squeeze() # torch.Size([2542888])
test_seeds = torch.tensor(test_seeds,dtype=torch.int32)
sampler = dgl.dataloading.MultiLayerNeighborSampler([10])

test_dataloader = dgl.dataloading.EdgeDataLoader(
graph, test_seeds, sampler,
negative_sampler=dgl.dataloading.negative_sampler.Uniform(1),
batch_size=test_mask.sum().item(),
shuffle=True,
drop_last=False,
num_workers=0)
#子图 目标节点id 找到目标节点
epoch_num = 600
learning_rate = 0.001
hidden_size = 256
out_dim = 32
lamda = 0
lamda_value = '0.9'
use_vardrop = 'False'
dropout = 0.9
model = GCN(graph.ndata['feature'].shape[1], hidden_size, out_dim, dropout, use_vardrop)
model_ID = 'ours'
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epoch_num, eta_min=1e-5)
#cora lr 1e-5 citeseer lr 5e-6

total_loss = []
total_acc = []
epoches = []
graph.ndata['embedding'] = torch.zeros(graph.num_nodes(),out_dim).cuda()
for epoch in range(epoch_num):
    loss_list = []
    for idx, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(dataloader):
        feature = graph.ndata['feature'][input_nodes.tolist()].cuda()
        node_embedding = model(blocks,feature)
        with torch.no_grad():
            graph.ndata['embedding'][blocks[1].dstdata['_ID'].tolist()] = node_embedding
        loss = Loss(node_embedding, pos_graph, neg_graph, negs_number)
        extra_loss = lamda * Radamacher_Regularization_dc(
        model, feature, 2)
        optimizer.zero_grad()
        (loss + extra_loss).backward()
        loss_list.append((loss + extra_loss).item())
        #loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
    CosineLR.step()
    total_loss.append(np.mean(loss_list))
    

    for idx, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(test_dataloader):
        feature = graph.ndata['embedding'][blocks[0].dstdata['_ID'].tolist()]
        acc = ACC(feature, pos_graph, neg_graph)
    total_acc.append(acc)
    info = ' epoch={}, | loss={},acc={}'.\
            format(epoch, np.mean(loss_list),acc)
    print(info)
    epoches.append(epoch)
info = 'model_ID={} | Dataset={}: acc: {} '.\
        format(model_ID, dataset, np.max(total_acc))
print(info)
#logs(model_ID+dataset+lamda_value+'_eval_GCN.log', info)

# 画出迭代与精度的曲线图
#plt.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文
plt.subplot(2,2,1)  #第一个图
plt.plot(epoches, total_loss, color='b', alpha=1, linewidth=1,label='loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.subplot(2,2,2)  #第二个图
plt.plot(epoches, total_acc, color='r', alpha=1, linewidth=1,label='train acc')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('test') 
plt.show()
plt.savefig('/home/anxy/zzh/Dropconnect/link prediction/figure/loss_acc.jpg') 
print('finish')