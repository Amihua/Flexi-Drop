# encoding: utf-8
import dgl
import numpy as np
from dgl.data import DGLDataset
import torch as th

filename = 'COX2'
file_path1 = '/home/anxy/zzh/Dropconnect/graph calssification/'+filename+'/'+filename+'_A.txt'
file_1 = open(file_path1, 'r', encoding='utf-8')
file_path2 = '/home/anxy/zzh/Dropconnect/graph calssification/'+filename+'/'+filename+'_graph_indicator.txt'
file_2 = open(file_path2, 'r', encoding='utf-8')
file_path3 = '/home/anxy/zzh/Dropconnect/graph calssification/'+filename+'/'+filename+'_graph_labels.txt'
file_3 = open(file_path3, 'r', encoding='utf-8')
file_path4 = '/home/anxy/zzh/Dropconnect/graph calssification/'+filename+'/'+filename+'_node_attributes.txt'
file_4 = open(file_path4, 'r', encoding='utf-8')
file_data1 = file_1.readlines()
file_data2 = file_2.readlines()
file_data3 = file_3.readlines()
file_data4 = file_4.readlines()

adjmatrix = np.zeros([len(file_data1),2])
m = 0
for lines in file_data1:
    lines = lines.rstrip()
    lines = lines.split(',')
    adjmatrix[m,:] = [int(lines[0]),int(lines[1])]
    m = m + 1

graphindex  = np.zeros(len(file_data2))
n = 0
for lines in file_data2:
    lines = lines.rstrip()
    graphindex[n] = int(lines)
    n = n + 1

label_list = []
for lines in file_data3:
    lines = lines.rstrip()
    label_list.append(int(lines))

nodefeature = np.zeros([len(file_data4),3])
t = 0
for lines in file_data4:
    lines = lines.rstrip()
    lines = lines.split(',')
    f_list = []
    for i in range(len(lines)):
        f_list.append(float(lines[i]))
    nodefeature[t,:] = f_list
    t = t + 1


adj = np.zeros([len(nodefeature),len(nodefeature)])
for index in adjmatrix:
    adj[int(index[0])-1,int(index[1])-1] = 1


k = 0
feature_list = []
adj_list = []
for i in range(int(max(graphindex))):
    tmp = graphindex[graphindex == i + 1]
    adj_list.append(th.tensor(adj[k:k+len(tmp),k:k+len(tmp)],dtype = th.int32))
    feature_list.append(nodefeature[k:k+len(tmp),:])
    k = k + len(tmp)

direction_1 = []
direction_2 = []
for i in range(len(adj_list)):
    j = 0
    in_nodes = th.tensor([])
    out_nodes = th.tensor([])
    for k in adj_list[i]:
        src = j * th.ones(adj_list[i].shape[0], dtype=th.int32)[k.to(th.bool)]  # 起始点
        dst = th.arange(0, adj_list[i].shape[0], dtype=th.int32)[k.to(th.bool)]  # 终点
        in_nodes = th.cat([in_nodes, src], dim=0)
        out_nodes = th.cat([out_nodes, dst], dim=0)
        j = j + 1  # j计算第j行
    in_nodes = th.cat([in_nodes,th.arange(0, adj_list[i].shape[0], dtype=th.int32)],dim=0)
    out_nodes = th.cat([out_nodes,th.arange(0, adj_list[i].shape[0], dtype=th.int32)],dim=0)
    direction_1.append(in_nodes.long())
    direction_2.append(out_nodes.long())
g=[]
for i in range(len(feature_list)):
    oneg = dgl.graph((direction_1[i], direction_2[i]))  # 如果具有最大ID的节点没有边，创建图时候指明节点数量
    oneg.ndata['feature'] = th.tensor(feature_list[i],dtype=th.float32)
    # oneg.add_edges(oneg.nodes(), oneg.nodes())
    g.append(oneg)    #生成一个一个小图，g是list
# for i in range(len(adj_list)):
#     if g[i].num_nodes() != feature_list[i].shape[0]:
#         print(i)
# A = 1
data_graph_path = '/home/anxy/zzh/Dropconnect/graph calssification/'+filename+'/'+filename+'.graph'
dgl.save_graphs(data_graph_path, g)



# in_nodes = th.tensor([])
# out_nodes = th.tensor([])
# for k in adj_list[37]:
#     src = j * th.ones(adj_list[37].shape[0], dtype=th.int32)[k.to(th.bool)]  # 起始点
#     dst = th.arange(0, adj_list[37].shape[0], dtype=th.int32)[k.to(th.bool)]  # 终点
#     in_nodes = th.cat([in_nodes, src], dim=0)
#     out_nodes = th.cat([out_nodes, dst], dim=0)
#     j = j + 1  # j计算第j行
# graph = dgl.graph((in_nodes, out_nodes))
#加一个自环






# class MyDataset(DGLDataset):

#     def __init__(self, name, raw_dir=None, force_reload=False, verbose=False):
#         super(MyDataset, self).__init__(name=name,
#                                         raw_dir=raw_dir,
#                                         force_reload=force_reload,
#                                         verbose=verbose)

#     def process(self):
#         textname = self.name 
#         self.graphs, self.labels = self.bulid_graph(textname)

#     def bulid_graph(self, filename):
#         file_path1 = '/home/anxy/zzh/Dropconnect/graph calssification/'+filename+'/'+filename+'_A.txt'
#         file_1 = open(file_path1, 'r', encoding='utf-8')
#         file_path2 = '/home/anxy/zzh/Dropconnect/graph calssification/'+filename+'/'+filename+'_graph_indicator.txt'
#         file_2 = open(file_path2, 'r', encoding='utf-8')
#         file_path3 = '/home/anxy/zzh/Dropconnect/graph calssification/'+filename+'/'+filename+'_graph_labels.txt'
#         file_3 = open(file_path3, 'r', encoding='utf-8')
#         file_path4 = '/home/anxy/zzh/Dropconnect/graph calssification/'+filename+'/'+filename+'_node_attributes.txt'
#         file_4 = open(file_path4, 'r', encoding='utf-8')
#         file_data1 = file_1.readlines()
#         file_data2 = file_2.readlines()
#         file_data3 = file_3.readlines()
#         file_data4 = file_4.readlines()

#         adjmatrix = np.zeros([len(file_data1),2])
#         m = 0
#         for lines in file_data1:
#             lines = lines.rstrip()
#             lines = lines.split(',')
#             adjmatrix[m,:] = [int(lines[0]),int(lines[1])]
#             m = m + 1

#         graphindex  = np.zeros(len(file_data2))
#         n = 0
#         for lines in file_data2:
#             lines = lines.rstrip()
#             graphindex[n] = int(lines)
#             n = n + 1

#         graphlabel = []
#         for lines in file_data3:
#             lines = lines.rstrip()
#             graphlabel.append(int(lines))

#         nodefeature = np.zeros([len(file_data4),3])
#         t = 0
#         for lines in file_data4:
#             lines = lines.rstrip()
#             lines = lines.split(',')
#             nodefeature[t,:] = [float(lines[0]),float(lines[1]),float(lines[2])]
#             t = t + 1

#         adj = np.zeros([len(nodefeature),len(nodefeature)])
#         for index in adjmatrix:
#             adj[int(index[0])-1,int(index[1])-1] = 1

#         k = 0
#         feature_list = []
#         label_list = graphlabel
#         adj_list = []
#         for i in range(int(max(graphindex))):
#             tmp = graphindex[graphindex == i + 1]
#             adj_list.append(adj[k:k+len(tmp),k:k+len(tmp)])
#             feature_list.append(nodefeature[k:k+len(tmp),:])
#             k = k + len(tmp)
#         direction_1 = []
#         direction_2 = []
#         for i in range(len(adj_list)):
#             j = 0
#             in_nodes = th.tensor([])
#             out_nodes = th.tensor([])
#             for k in adj_list[i]:
#                 src = j * th.ones(adj_list[i].shape[0], dtype=th.int32)[k.to(th.bool)]  # 起始点
#                 dst = th.arange(0, adj_list[i].shape[0], dtype=th.int32)[k.to(th.bool)]  # 终点
#                 in_nodes = th.cat([in_nodes, src], dim=0)
#                 out_nodes = th.cat([out_nodes, dst], dim=0)
#                 j = j + 1  # j计算第j行
#             direction_1.append(in_nodes.long())
#             direction_2.append(out_nodes.long())
#         graph_list=[]
#         for i in range(len(feature_list)):
#             oneg = dgl.graph((direction_1[i], direction_2[i]))  # 如果具有最大ID的节点没有边，创建图时候指明节点数量
#             oneg.ndata['feature'] = th.tensor(feature_list[i])
#             oneg.add_edges(oneg.nodes(), oneg.nodes())
#             graph_list.append(oneg)    #生成一个一个小图，g是list

#         return graph_list, label_list

#     def __getitem__(self, idx):
#         return self.graphs[idx], self.labels[idx]

#     def __len__(self):
#         return len(self.graphs)