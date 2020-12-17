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
        graph_n_n = graph_n_n[node_x.reshape(-1).cpu().numpy(),  :]
        t = time.time()
        graph_n_n  =  sparse_mx_to_torch_sparse_tensor(graph_n_n).float()
        t = time.time()
        adj_x_n = graph_n_n
        adj_x_n = adj_x_n.cuda()
        features_n_f = features_n_f.cuda()
        t = time.time()
        x_f_self = torch.index_select(features_n_f,0,node_x.reshape(-1))
        t = time.time()
        output_x_f = torch.spmm(adj_x_n, features_n_f) 
        t = time.time()
        output = torch.cat([x_f_self, output_x_f], dim = 1)
        output = self.full(output)
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

    
class WeightLayer(torch.nn.Module):
    def __init__(self, args):
        super(WeightLayer, self).__init__()
        self.args = args
        self.linear = torch.nn.Linear(self.args.feature_dim, 1)
        torch.nn.init.xavier_uniform_(self.linear.weight)
    
    def forward(self, x):
        return self.linear(x)
    

class Walk(torch.nn.Module):
    def __init__(self, args):
        super(Walk, self).__init__()
        self.args = args
        self.linear = torch.nn.Linear(self.args.feature_dim*2, 1)
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, walk_times, adj_sparse, train_index, batch_size, features):
        nodes_num = len(train_index)
        walks = torch.zeros(nodes_num, walk_times+1).long().cuda()
        batches = create_batches_forList(train_index, batch_size, True)
        i=0
        candi_node = 0
        indexes = np.arange(0, adj_sparse.shape[0])
        for batch in batches:
            walks[i*batch_size : i*batch_size + len(batch), 0] = torch.tensor(batch).cuda()
            node_adj = adj_sparse[batch.cpu().numpy()]
            candi_node = sparse_mx_to_torch_sparse_tensor(node_adj).cuda()
            chosen_node = torch.zeros(len(batch), adj_sparse.shape[0]).cuda()
            for id in range(len(batch)):
                chosen_node[id][batch[id]] = 1.
            candi_node = ((- chosen_node + candi_node)> 0.0).float()
            for x in range(candi_node.shape[0]):
                if candi_node[x].sum() < 0:
                    print("x = {}".format(x))
                if candi_node[x].sum()==0:
                    candi_node[x][batch[x]] = 1.
                else:
                    x_feat = features[batch[x]]
                    x_feat = torch.reshape(x_feat, shape=(1, x_feat.shape[0]))
                    batch_x_list = np.array([batch[x]], dtype=np.int32)
                    tmp_adj = adj_sparse[batch_x_list]
                    tmp_adj = np.array(tmp_adj.toarray()[0], dtype=np.bool)
                    tmp_adj = indexes[tmp_adj]
                    for y in tmp_adj:
                        y_feat = features[y]
                        y_feat = torch.reshape(y_feat, shape=(1, y_feat.shape[0]))
                        xy_feat = torch.cat([x_feat, y_feat], 1)
                        mm_value = self.linear(xy_feat)[0][0]
                        if mm_value >= 0:
                            candi_node[x][y] = mm_value
                        else:
                            candi_node[x][y] = 0.
            for x in range(candi_node.shape[0]):
                if candi_node[x].sum()==0:
                    candi_node[x][batch[x]] = 1.

            for walk_id in range(walk_times):
                p = candi_node
                new_node = torch.multinomial(p,1).squeeze(1)
                walks[i*batch_size : i*batch_size + len(batch), walk_id+1] = new_node
                for id in range(len(batch)):
                    chosen_node[id][new_node[id]] = 1.
                candi_node = candi_node - chosen_node + sparse_mx_to_torch_sparse_tensor(adj_sparse[new_node.cpu()]).cuda()
                candi_node = (candi_node> 0.0).float()

            i+=1
        return walks
