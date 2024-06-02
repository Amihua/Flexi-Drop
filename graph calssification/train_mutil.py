import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import dgl
import numpy as np
import torch
import math
import random
import matplotlib.pyplot as plt
from dgl.dataloading import GraphDataLoader
import time
import torch.nn.functional as F
from torch.utils.data import Dataset
from model import Graphsage, VariationalDropout,SAGEConv


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def logs(log_file, content):
    with open(os.path.join('/home/anxy/zzh/Dropconnect/graph calssification/result', log_file), mode='a') as f:
        f.write(content + '\t\n')

def evaluate(model, g, features, labels):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

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
        elif isinstance(layer, SAGEConv):
            loss *= 2 * torch.max(torch.abs(layer.fc.weight))

            k_new, k_old = layer.fc.weight.shape

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
        elif isinstance(layer, SAGEConv):
            loss *= 2 * torch.max(torch.norm(layer.apply_func.weight, p = 2, dim = 0))
            k_new, k_old = layer.apply_func.weight.shape
            loss *= np.sqrt(k_new + k_old) / k_new

    return loss


dataset = 'ENZYMES'
file_path = '/home/anxy/zzh/Dropconnect/graph calssification/'+dataset+'/'+dataset+'_graph_labels.txt'
file = open(file_path, 'r', encoding='utf-8')
file_data = file.readlines()
label_list = []
for lines in file_data:
    lines = lines.rstrip()
    label_list.append(int(lines))
index0 = [index for index,label in enumerate(label_list) if label == 0]
index1 = [index for index,label in enumerate(label_list) if label == 1]
index2 = [index for index,label in enumerate(label_list) if label == 2]
index3 = [index for index,label in enumerate(label_list) if label == 3]
index4 = [index for index,label in enumerate(label_list) if label == 4]
index5 = [index for index,label in enumerate(label_list) if label == 5]
random.shuffle(index0)
random.shuffle(index1)
random.shuffle(index2)
random.shuffle(index3)
random.shuffle(index4)
random.shuffle(index5)
train_index0 = index0[:math.ceil(len(index0)*0.8)]
test_index0 = index0[math.ceil(len(index0)*0.8):]
train_index1 = index1[:math.ceil(len(index1)*0.8)]
test_index1 = index1[math.ceil(len(index1)*0.8):]
train_index2 = index2[:math.ceil(len(index2)*0.8)]
test_index2 = index2[math.ceil(len(index2)*0.8):]
train_index3 = index1[:math.ceil(len(index3)*0.8)]
test_index3 = index1[math.ceil(len(index3)*0.8):]
train_index4 = index0[:math.ceil(len(index4)*0.8)]
test_index4 = index0[math.ceil(len(index4)*0.8):]
train_index5 = index1[:math.ceil(len(index5)*0.8)]
test_index5 = index1[math.ceil(len(index5)*0.8):]
train_index = train_index0 + train_index1 + train_index2 + train_index3 + train_index4 + train_index5
test_index = test_index0 + test_index1 + test_index2 + test_index3 + test_index4 + test_index5
graph_path = '/home/anxy/zzh/Dropconnect/graph calssification/'+ dataset +'/'+ dataset +'.graph'
graph_list, _ = dgl.load_graphs(graph_path)

train_label = []
train_graph = []
for index in train_index:
    train_label.append(label_list[index])
    train_graph.append(graph_list[index])
train_label = torch.tensor(train_label)
train_label = train_label.long().to(device)

test_label = []
test_graph = []
for index in test_index:
    test_label.append(label_list[index])
    test_graph.append(graph_list[index])
test_label = torch.tensor(test_label,dtype=torch.int32).to(device)
test_label= test_label.long().to(device)

#定义类作为对象输入GraphDataLoader中
class TensorDataset(Dataset):
    """
    TensorDataset继承Dataset, 重载了__init__(), __getitem__(), __len__()
    实现将一组Tensor数据对封装成Tensor数据集
    能够通过index得到数据集的数据，能够通过len，得到数据集大小
    """
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return len(self.data_tensor)

train_dataset = TensorDataset(train_graph, train_label)  # 将数据封装成Dataset，实例化类，就可使用索引调用数据tensor_dataset[i]
test_dataset = TensorDataset(test_graph,test_label)

#构建模型
dataloader = GraphDataLoader(dataset=train_dataset,batch_size=16, shuffle=True)
train_dataloader = GraphDataLoader(dataset=train_dataset,batch_size=len(train_graph), shuffle=True)
test_dataloader = GraphDataLoader(dataset=test_dataset,batch_size =len(test_graph),shuffle=False)

epoch_num = 300
learning_rate = 0.001
hidden_size = 256
out_dim = 32
dropout = 0.9
use_vardrop = 'True'
model = Graphsage(graph_list[0].ndata['feature'].shape[1], hidden_size, out_dim, torch.max(train_label).item()+1, dropout, use_vardrop)
model_ID = 'drop'
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epoch_num, eta_min=1e-5)


total_loss = []
total_acc = []
epoches = []
for epoch in range(epoch_num):
    loss_list = []
    for batched_graph,batch_label in dataloader:
        batched_graph = batched_graph.to(device)
        feature = batched_graph.ndata['feature'].to(device)
        logits = model(batched_graph,feature).to(device)
        loss = F.cross_entropy(logits, batch_label)
        extra_loss = 0.6 * Radamacher_Regularization_dc(model, feature, torch.max(train_label).item()+1)
        optimizer.zero_grad()
        (loss + extra_loss).backward()
        #loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
    CosineLR.step()

    for train_graph,train_label in train_dataloader:
        train_graph = train_graph.to(device)
        train_feats = train_graph.ndata['feature'].to(device)
    trainloss = F.cross_entropy(model(train_graph,train_feats).to(device),train_label).to(device)
    extra_loss = 0.6 * Radamacher_Regularization_dc(model, feature, torch.max(train_label).item()+1)
    totalloss= (trainloss + extra_loss).detach().cpu().numpy()
    #totalloss= (trainloss).detach().cpu().numpy()
    total_loss.append(totalloss)

    for test_graph, test_label in test_dataloader:
        test_graph = test_graph.to(device)
        test_feats = test_graph.ndata['feature'].to(device)
    acc = evaluate(model, test_graph,test_feats, test_label)
    total_acc.append(acc)
    info = ' epoch={}, | train_loss={},acc={}'.\
            format(epoch, totalloss, acc)
    print(info)
    epoches.append(epoch)

info = 'model_ID={} | Dataset={}: acc: {} '.\
        format(model_ID, dataset, np.max(total_acc))
print(info)
logs(model_ID+dataset+'_eval_GIN.log', info)


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
plt.savefig('/home/anxy/zzh/Dropconnect/graph calssification/figure/loss_acc.jpg') 
print('finish')