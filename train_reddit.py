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

class Train_Model_reddit():
    def __init__(self, args):
        self.args = args
        self.sparse_adj, self.sparse_adj_train, self.features, self.train_feature, self.labels, self.train_labels, self.id_train, self.id_valid, self.id_test, num_labels = load_data2(self.args.dataset,True)
        self.sample_batch_size = 256
        print('train walk')
        name = 'seed='+str(self.args.seed)+'_walk='+str(self.args.time)
        self.walks_train = pre_sample(self.args.time, self.sparse_adj_train, self.id_train, self.sample_batch_size, 'tain_'+name,self.args.way,False)
        print('valid walk')
        self.walks_valid = pre_sample(self.args.time, self.sparse_adj, self.id_valid, self.sample_batch_size,'valid_'+name,self.args.way,False)
        self.sparse_adj.setdiag(1.0) 
        self.sparse_adj_train.setdiag(1.0) 
        #self.nor_graph = normalize(self.nor_graph, norm='l1', axis=1)
        self.nor_graph = normalize(self.sparse_adj, norm='l1', axis=1)
        self.nor_graph_train = normalize(self.sparse_adj_train, norm='l1', axis=1)
        self.args.feature_dim = self.features.shape[1]
        self.args.num_nodes = self.features.shape[0]
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
        ave_epoch_time = 0
        ave_batch_time = 0
        total_start = time.time()
        for epoch in tqdm(range(self.args.epochs)):
            self.model.train()
            self.epoch_loss = 0.0
            self.acc_score = 0.0
            self.nodes_processed = 0.0
            batch_range = len(batches)
            batch_time = 0
            epoch_start = time.time()
            
            for batch in range(batch_range):
                label = torch.index_select(self.train_labels, 0, batches[batch][:,0].view(-1))
                start = time.time()
                self.epoch_loss = self.epoch_loss + self.process_batch(label, batches[batch], self.train_feature, self.nor_graph_train)
                batch_time +=  time.time() - start
            epoch_end = time.time()
            ave_epoch_time += epoch_end - epoch_start
            batch_time = batch_time / batch_range
            ave_batch_time += batch_time
            self.model.eval()
            valid_loss = 0.0
            valid_acc = 0.0
            
            for batch in valid_batch:
                label = torch.index_select(self.labels, 0, batch[:,0].view(-1))
                loss_node, acc, label_predict = self.process_node(label, batch, self.features, self.nor_graph)
                valid_loss += loss_node.item()
                valid_acc += acc.item()
                
            valid_loss = round(valid_loss*1000/len(self.id_valid), 4)
            valid_acc = round(valid_acc/len(self.id_valid), 4)
            self.acc_score = round(self.acc_score/len(self.id_train), 4)
            loss_score = round(self.epoch_loss*1000/len(self.id_train), 4)
            if epoch % 1 == 0:
                print("epoch",epoch,"loss_train:",loss_score,"acc_train:",self.acc_score,'||',"loss_valid:",valid_loss,"acc_valid:",valid_acc, '|| batch time:', round(batch_time, 4), '||epoch time:', round(epoch_end - epoch_start, 4))
            if  valid_loss < self.best_loss :
                self.best_loss = valid_loss
                torch.save(self.model.state_dict(),self.args.save_path_loss+"best_model.pt")
                stop_count = 0
                self.best_loss_epoch = epoch
            else:
                stop_count += 1
                if stop_count == self.args.patience:
                    print(self.args.patience, "times no decrease")
                    #print(self.total_time / (epoch+1))
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
        #now1 = torch.cuda.memory_allocated()
        self.acc_score += acc.item()
        batch_loss.backward()
        #now2 = torch.cuda.memory_allocated()
        self.optimizer.step()
        #now3 = torch.cuda.memory_allocated()
        #print('now1',now1-pre,'now2',now2-pre,'now3',now3-pre)
        return batch_loss.item()
    
