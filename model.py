from sub_net import *
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

    def forward(self, node, graph, features):
        A_next = self.aggregator(node, graph, features)
        res = self.mean(A_next)
        return res