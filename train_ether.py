from model import Model_Layer
import numpy as np
import networkx as nx
from utils import *
import random
import torch
from tqdm import tqdm_notebook as tqdm
import time
import scipy.sparse as sparse
from sklearn.preprocessing import normalize

def pre_sample_for_ether(walk_times,adj_sparse, train_index, batch_size, save_name, do_walk):
    if do_walk:
        nodes_num = len(train_index)
        walks = torch.zeros(nodes_num, walk_times+1).long().cuda()
        batches = create_batches_forList(train_index, batch_size, True)
        i=0
        degrees = torch.tensor(adj_sparse.sum(1)).view(-1).cuda()
        candi_node = 0
        for batch in batches:
            walks[i*batch_size : i*batch_size + len(batch), 0] = torch.tensor(batch).cuda()
            candi_node = sparse_mx_to_torch_sparse_tensor(adj_sparse[batch.cpu().numpy()]).cuda()
            chosen_node = torch.zeros(len(batch), adj_sparse.shape[0]).cuda()
            for id in range(len(batch)):
                chosen_node[id][batch[id]] = 1.
            candi_node = ((- chosen_node + candi_node)> 0.0).float()
            for walk_id in range(walk_times):
                for x in range(candi_node.shape[0]):
                    if candi_node[x].sum()==0:
                        candi_node[x][batch[x]] = 1.
               
                p = candi_node * degrees
                new_node = torch.multinomial(p,1).squeeze(1)
                walks[i*batch_size : i*batch_size + len(batch), walk_id+1] = new_node
                for id in range(len(batch)):
                    chosen_node[id][new_node[id]] = 1.
                candi_node = candi_node - chosen_node + sparse_mx_to_torch_sparse_tensor(adj_sparse[new_node.cpu()]).cuda()
                candi_node = (candi_node> 0.0).float()
                
            i+=1
        torch.cuda.empty_cache()
        result1 = np.array(walks.cpu())
        io.savemat(save_name+'.mat',{save_name:result1})
        return walks
    else:
        walks = io.loadmat(save_name+'.mat')
#         print("walks",walks)
        return torch.tensor(walks[save_name]).cuda()

class Train_Model_Ether():
    def __init__(self, args):
        self.args = args
        self.sparse_adj, self.sparse_adj_train,self.sparse_adj_train_all, self.features, self.train_feature, self.train_feature_all,self.labels, self.train_labels, self.id_train, self.id_valid, self.id_test, num_labels = Origin_load_ether_data()
        self.sample_batch_size = 256
        print('train walk')
        name = 'seed='+str(self.args.seed)+'_walk='+str(self.args.time)
        self.walks_train = pre_sample_for_ether(self.args.time, self.sparse_adj_train, self.id_train, self.sample_batch_size, './walks_ether/train_'+name,False)
        print('valid walk')
        self.walks_valid = pre_sample_for_ether(self.args.time, self.sparse_adj, self.id_valid, self.sample_batch_size,'./walks_ether/valid_'+name,False)
        self.sparse_adj.setdiag(1.0) 
        self.sparse_adj_train.setdiag(1.0) 
        #self.nor_graph = normalize(self.nor_graph, norm='l1', axis=1)
        self.nor_graph = normalize(self.sparse_adj, norm='l1', axis=1)
        self.nor_graph_train = normalize(self.sparse_adj_train, norm='l1', axis=1)
        self.nor_graph_train_all = normalize(self.sparse_adj_train_all, norm='l1', axis=1)
#         self.nor_graph_train_all = self.nor_graph_train_all.cuda()
#         print("self.nor_graph_train_all",self.nor_graph_train_all)
#         print("self.nor_graph_train_all_type",type(self.nor_graph_train_all))
        self.args.feature_dim = self.features.shape[1]
#         self.args.num_nodes = self.features.shape[0]
        self.args.num_nodes = num_labels
        self.args.num_labels = num_labels
        self.model = Model_Layer(self.args).cuda()
        print('cuda ready')
        #self.model = torch.nn.DataParallel(Model_Layer(self.args),device_ids=[0,1,2,3])
        self.logs = create_logs(self.args)
        self.best_loss = 1000
        self.best_acc = 0
        self.best_loss_both = 1000
        self.best_acc_both = 0
        self.stop_count = 0
        self.best_loss_epoch = -1
        self.best_both_epoch = -1
        self.best_acc_epoch  = -1
        self.epoch_idx = 0
        self.total_time = 0.0
        
    def fit(self):
        print("\nTraining started.\n")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.args.learning_rate, weight_decay = self.args.weight_decay)
        self.optimizer.zero_grad()
        batches = create_batches_forWalk(self.walks_train , self.args.batch_size)
        valid_batch = create_batches_forWalk(self.walks_valid , self.args.batch_size)
        self.id_train = list(self.id_train)
        for epoch in tqdm(range(self.args.epochs)):
            self.model.train()
            self.epoch_loss = 0.0
            self.acc_score = 0.0
            self.nodes_processed = 0.0
            batch_range = len(batches)
            
            for batch in range(batch_range):
                label = torch.index_select(self.train_labels, 0, batches[batch][:,0].view(-1))
                self.epoch_loss = self.epoch_loss + self.process_batch(label, batches[batch], self.train_feature_all, self.nor_graph_train_all)
                print("epoch_loss",self.epoch_loss)
            self.model.eval()
            valid_loss = 0.0
            valid_acc = 0.0
            
            for batch in valid_batch:
                label = torch.index_select(self.labels, 0, batch[:,0].view(-1))
                loss_node, acc, label_predict = self.process_node(label, batch, self.features, self.nor_graph)
                print("loss_node",loss_node,"acc:",acc,"label_predict:",label_predict)
                valid_loss += loss_node.item()
                valid_acc += acc.item()
                
            valid_loss = round(valid_loss*1000/len(self.id_valid), 4)
            valid_acc = round(valid_acc/len(self.id_valid), 4)
            self.acc_score = round(self.acc_score/len(self.id_train), 4)
            loss_score = round(self.epoch_loss*1000/len(self.id_train), 4)
            if epoch % 1 == 0:
                print("epoch",epoch,"loss_train:",loss_score,"acc_train:",self.acc_score,'||',"loss_valid:",valid_loss,"acc_valid:",valid_acc)
            if  valid_loss < self.best_loss :
                self.best_loss = valid_loss
                torch.save(self.model.state_dict(),self.args.save_path_loss+"best_model.pt")
                stop_count = 0
                self.best_loss_epoch = epoch
            else:
                stop_count += 1
                if stop_count == self.args.patience:
                    print(self.args.patience, "times no decrease")
                    return round(ave_epoch_time/(epoch+1), 4), round(ave_batch_time / (epoch+1), 4), time.time()-total_start
        print('Max epoches reaches!')
        return round(ave_epoch_time/(epoch+1), 4), round(ave_batch_time / (epoch+1), 4), time.time()-total_start
            
    def evaluation(self):
        loss_result = torch.zeros(len(self.id_test), self.args.eva_times, dtype=torch.long)
        acc_result = torch.zeros(len(self.id_test), self.args.eva_times, dtype=torch.long)
        loss_acc = 0.0
        acc_acc = 0.0
        #print('test walk')
        name = 'seed='+str(self.args.seed)+'_walk='+str(self.args.time)
        self.walks_test = pre_sample(self.args.time, self.sparse_adj, self.id_test, self.sample_batch_size,'test_'+name, self.args.way,False)
        test_batch = create_batches_forWalk(self.walks_test, self.args.batch_size)
        
        self.model.eval()
        self.model.load_state_dict(torch.load(self.args.save_path_loss+"best_model.pt"))
        for batch in test_batch:
            label = torch.index_select(self.labels, 0, batch[:,0].view(-1))
            loss_node, acc, label_predict = self.process_node(label, batch, self.features, self.nor_graph)
            loss_acc += acc.item()
        
        loss_acc = round(loss_acc/len(self.id_test), 4)
        print("loss acc:", loss_acc, 'load epoch:', self.best_loss_epoch)
        return  loss_acc, acc_acc
    def process_node(self,label, node, feature, graph):
        prediction = self.model(node, graph, feature)
        prediction_loss = calculate_predictive_loss(label, prediction)
        acc, label_pre = calculate_reward(label, prediction)
        return prediction_loss, acc, label_pre
    
    def process_batch(self,label,batch, feature, graph):
        self.optimizer.zero_grad()
        pre = torch.cuda.memory_allocated()
        batch_loss, acc, label_pre = self.process_node(label, batch, feature, graph)
        self.acc_score += acc.item()
        batch_loss.backward()
        self.optimizer.step()
        return batch_loss.item()
    
