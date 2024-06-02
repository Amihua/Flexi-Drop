# encoding: utf-8
import json
import sys
import os
import torch
import dgl
import numpy as np
import pickle
import math

file_path1 = '/home/anxy/zzh/Dropconnect/link prediction/pubmed/Pubmed-Diabetes.DIRECTED.cites.tab'
file_1 = open(file_path1, 'r', encoding='utf-8')
file_path3 = '/home/anxy/zzh/Dropconnect/link prediction/pubmed/Pubmed-Diabetes.NODE.paper.tab'
file_3 = open(file_path3, 'r', encoding='utf-8')
file_data1 = file_1.readlines()
file_data3 = file_3.readlines()
data = []
for lines in file_data3:
    lines = lines.rstrip()  # 删去行末的换行符
    lines = lines.split('\t')  # 按空格‘\t'分割数据
    lines = list(map(lambda x: x.lstrip(), lines))  # 去掉数据前的空格
    # lines = list(map(int, lines))  # 转换为张量不能含有str数据，但float默认保留4位小数
    data.append(lines)
dataset  = data[2:]
ID = []
for i in range(len(dataset)):
    ID.append(int(dataset[i][0]))
edgeid = []
for lines in file_data1:
    lines = lines.rstrip()  # 删去行末的换行符
    lines = lines.split('\t')  # 按空格‘\t'分割数据
    lines = list(map(lambda x: x.lstrip(), lines))  # 去掉数据前的空格
    edgeid.append(lines)
edgeid = edgeid[2:]
edge_matrix = np.zeros([len(edgeid),2])
for i in range(len(edge_matrix)):
    edge_matrix[i,0] = int(edgeid[i][1])
    edge_matrix[i,1] = int(edgeid[i][3])
for (k,i) in enumerate(ID):
    edge_matrix[edge_matrix == i] = k

num_out_node = edge_matrix[:,0]
num_in_node = edge_matrix[:,1]
out_node = torch.tensor(num_out_node,dtype=torch.int64)
in_node = torch.tensor(num_in_node,dtype=torch.int64)
out_id = torch.cat([out_node,in_node])
in_id = torch.cat([in_node,out_node])
data_graph = dgl.graph((out_id,in_id))
data_graph.ndata['feature'] = torch.randn(data_graph.num_nodes(),500)
edge_id = torch.arange(data_graph.num_edges())
sample_edge = torch.utils.data.RandomSampler(edge_id,replacement = False, num_samples = None, generator = None)
sample_edges = torch.tensor([i for i in sample_edge])
train_mask_id = sample_edges[:math.ceil(len(sample_edges)*0.8)]
test_mask_id = sample_edges[math.ceil(len(sample_edges)*0.8):]
train_mask = torch.zeros(data_graph.num_edges())
train_mask[train_mask_id] = 1
test_mask = torch.zeros(data_graph.num_edges())
test_mask[test_mask_id] = 1
data_graph.edata['train_mask'] = train_mask.to(torch.bool)
data_graph.edata['test_mask'] = test_mask.to(torch.bool)
data_graph_path = '/home/anxy/zzh/Dropconnect/link prediction/pubmed/pubmed.graph'
dgl.save_graphs(data_graph_path, data_graph)