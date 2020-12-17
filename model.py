from sub_net import *
from utils import *
import numpy as np
import networkx as nx
import random
import torch
import os
import time
class Model_Layer(torch.nn.Module):
    def __init__(self, args):
        super(Model_Layer, self).__init__()
        self.args = args
        self.aggregator = Aggregator(args)
        self.mean = Mean(args)

    def forward(self, node, graph, features, dropout=0.):
        A_next = self.aggregator(node, graph, features)
        res = self.mean(A_next)
        return res


class AttnWeight(torch.nn.Module):
    def __init__(self, args):
        super(AttnWeight, self).__init__()
        self.args = args
        self.aggregator = Aggregator(args)
        self.mean = Mean(args)
        self.walk = Walk(args)

    def forward(self, graph, features, walk_times, adj_sparse, train_index, batch_size):
        walks = self.walk(walk_times, adj_sparse, train_index, batch_size, features)
        A_next = self.aggregator(walks, graph, features)
        prediction = self.mean(A_next)
        return prediction

class MultiHeadLayer(torch.nn.Module):
    def __init__(self, args, head_nums=1):
        super(MultiHeadLayer, self).__init__()
        print("multi head")
        self.args = args
        self.head_nums = head_nums
        self.aggregators, self.means = [], []
        for i in range(head_nums):
            aggregator, mean = Aggregator(args), Mean(args)
            setattr(self, "aggregator_" + str(i), aggregator)
            setattr(self, "mean_" + str(i), mean)
            self.aggregators.append(aggregator)
            self.means.append(mean)

    def forward(self, node, graph, features, dropout=0.):
        result = None
        m = torch.nn.Dropout(p=dropout)
        for i in range(self.head_nums):
            dropout_features = m(features)
            A_next = self.aggregators[i](node, graph, dropout_features)
            res = self.means[i](A_next)
            if result is None:
                result = res
            else:
                result = torch.add(result, res)
        return result/self.head_nums
