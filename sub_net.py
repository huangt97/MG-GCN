import numpy as np
import networkx as nx
import random
import torch
import time
from utils import *

    
class Mean(torch.nn.Module):
    def __init__(self, args):
        super(Mean, self).__init__()
        self.args = args
        self.full = torch.nn.Linear(self.args.hidden*2, self.args.num_labels)
        torch.nn.init.xavier_uniform_(self.full.weight)
        self.args = args
        
    def forward(self, nodes_b_seq_h):
        res_b_h = torch.mean(nodes_b_seq_h[:,1:,:], 1)
        #res_b_h = nodes_b_seq_h.squeeze(1)
        res_cat = torch.cat([nodes_b_seq_h[:,0,:], res_b_h], dim = 1)
        res = torch.nn.functional.log_softmax(self.full(res_cat))
        return res   

class Aggregator(torch.nn.Module):
    def __init__(self, args):
        super(Aggregator, self).__init__()
        self.args = args
        self.full = torch.nn.Linear(self.args.feature_dim*2, self.args.hidden)
        torch.nn.init.xavier_uniform_(self.full.weight)
    def forward(self, node_x, graph_n_n, features_n_f):
#         print("type of graph_n_n:",type(graph_n_n),graph_n_n.shape)
        graph_n_n = graph_n_n[node_x.view(-1).cpu().numpy(),  :]
#         print("type of graph_n_n:",type(graph_n_n),graph_n_n.shape)
#         print("type of features_n_f:",type(features_n_f),features_n_f.shape)
#         print("type of node_x:",type(node_x.view(-1)),node_x.view(-1).shape)
        t = time.time()
        graph_n_n  =  sparse_mx_to_torch_sparse_tensor(graph_n_n).float()
#         print("type of graph_n_n:",type(graph_n_n),graph_n_n.shape)
#         print("first index:",time.time()-t)
#         print("type of graph_n_n:",type(graph_n_n),graph_n_n.shape)
        t = time.time()
        adj_x_n = graph_n_n
#         adj_x_n  = torch.index_select(graph_n_n,0,node_x.view(-1))
#         print("type of adj_x_n:",type(adj_x_n),adj_x_n.shape)
        adj_x_n = adj_x_n.cuda()
        features_n_f = features_n_f.cuda()
#         print("type of features_n_f:",type(features_n_f),features_n_f.shape)
#         print("secord index:",time.time()-t)
#         print("adj_x_n",adj_x_n)
        t = time.time()
        x_f_self = torch.index_select(features_n_f,0,node_x.view(-1))
#         print("x_f_self",x_f_self)
#         print("features_n_f",features_n_f)
        t = time.time()
        output_x_f = torch.spmm(adj_x_n, features_n_f) 
#         print("first matrix finished!",time.time()-t)
        t = time.time()
        output = torch.cat([x_f_self, output_x_f], dim = 1)
        output =  self.full(output)
        output = torch.nn.functional.leaky_relu(output)
        output = output.view(node_x.shape[0], self.args.time+1, self.args.hidden)
        return output
    
class DownStream(torch.nn.Module):
    def __init__(self, args):
        super(DownStream, self).__init__()
        self.args = args
        self.full = torch.nn.Linear(self.args.hidden, self.args.num_labels)
        torch.nn.init.xavier_uniform_(self.full.weight)
        
    def forward(self, H_output):
        #full_out = self.full(H_output)
        output = torch.nn.functional.log_softmax(H_output,dim = 1)
        return output    
