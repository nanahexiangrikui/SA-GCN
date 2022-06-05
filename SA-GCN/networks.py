import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from layers import SAGPool
import numpy as np


class Net(torch.nn.Module):
    def __init__(self,args):
        super(Net, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_nodes = args.num_nodes
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.pool1 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        self.conv2 = GCNConv(self.num_nodes, self.nhid)
        self.pool2 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.pool3 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        self.conv4 = GCNConv(self.nhid, self.nhid)
        self.pool4 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        self.conv5 = GCNConv(self.nhid, self.nhid)
        self.pool5 = SAGPool(self.nhid, ratio=self.pooling_ratio)

        self.lin1 = torch.nn.Linear(self.nhid*4, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid)
        self.lin3 = torch.nn.Linear(self.nhid, self. num_classes)


    def forward(self, data):
        x, edge_index, batch, ss = data.x, data.edge1_index, data.batch, data.s
        # x为原始数据的均值和方差特征矩阵，将x送入图神经网络和池化
        x = F.relu(self.conv1(x, edge_index))
        print(x)
        x, edge_index2, _, batch, _ = self.pool1(x, edge_index, None, batch)
        print(x)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x, edge_index, batch, ss = data.x, data.edge2_index, data.batch, data.s
        # ss为功能连接矩阵，将ss送入图神经网络和池化
        ss = F.relu(self.conv2(ss, edge_index))
        ss, _, _, batch, _ = self.pool2(ss, edge_index, None, batch)
        x2 = torch.cat([gmp(ss, batch), gap(ss, batch)], dim=1)

        #将两种结果连接在一起
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.lin1(x))
        #print(x.shape)
        #x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x