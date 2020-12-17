import pickle
import scipy.sparse as sp
import numpy as np
from collections import defaultdict


# Load data
def load_data(dataset_dir, labeled=False):
    adj = sp.load_npz(dataset_dir + "ether_adj.npz")
    data = np.load(dataset_dir + "ether_fastGCN.npz")
    adj, feat, y_train, y_val, y_test, train_index, val_index, test_index = \
        adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], \
        data['train_index'], data['val_index'], data['test_index']  # , data['train_target']
    feat = np.nan_to_num(feat)
    used_labels = np.concatenate([y_train, y_val, y_test])
    used_nodes = np.concatenate([train_index, val_index, test_index])
    labels = np.zeros((adj.shape[0], used_labels.max() + 1))
    train_mask, val_mask, test_mask = np.zeros(labels.shape[0]), np.zeros(labels.shape[0]), np.zeros(labels.shape[0])
    # label
    for index, label in enumerate(y_train):
        labels[train_index[index]][label] = 1
        train_mask[train_index[index]] = 1
    for index, label in enumerate(y_val):
        labels[val_index[index]][label] = 1
        val_mask[val_index[index]] = 1
    for index, label in enumerate(y_test):
        labels[test_index[index]][label] = 1
        test_mask[test_index[index]] = 1
    train_mask = np.array(train_mask, dtype=np.bool)
    val_mask = np.array(val_mask, dtype=np.bool)
    test_mask = np.array(test_mask, dtype=np.bool)
    y_train, y_val, y_test = np.zeros(labels.shape), np.zeros(labels.shape), np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    if labeled:
        # only labeled node
        csr_adj = adj.tocsr()
        small_adj = sp.lil_matrix((used_labels.shape[0], used_labels.shape[0]))
        small_feat = sp.lil_matrix((used_labels.shape[0], feat.shape[1]))
        small_y_train, small_y_val, small_y_test = \
            np.zeros((small_feat.shape[0], y_train.shape[1])), np.zeros((small_feat.shape[0], y_train.shape[1])), \
            np.zeros((small_feat.shape[0], y_train.shape[1]))
        small_train_mask, small_val_mask, small_test_mask = \
            np.zeros((small_feat.shape[0],)), np.zeros((small_feat.shape[0],)), np.zeros((small_feat.shape[0],))
        for node in used_nodes:
            nbr_nodes = csr_adj[node, :].toarray()
            small_feat[node, :] = feat[node, :]
            small_y_train[node], small_y_val[node], small_y_test[node] = \
                y_train[node], y_val[node], y_test[node]
            small_train_mask[node], small_val_mask[node], small_test_mask[node] = \
                train_mask[node], val_mask[node], test_mask[node]
            for nbr in used_nodes:
                small_adj[node, nbr] = nbr_nodes[0][nbr]
        small_train_mask = np.array(small_train_mask, dtype=np.bool)
        small_val_mask = np.array(small_val_mask, dtype=np.bool)
        small_test_mask = np.array(small_test_mask, dtype=np.bool)
        return small_adj, small_feat, small_y_train, small_y_val, small_y_test, small_train_mask, small_val_mask, small_test_mask
    feat = sp.lil_matrix(feat)
    return adj, feat, y_train, y_val, y_test, train_mask, val_mask, test_mask


def statistics_data(dataset_dir):
    adj = sp.load_npz(dataset_dir + "ether_adj.npz")
    data = np.load(dataset_dir + "ether.npz")
    adj = adj.tolil()
    adj_dict = defaultdict(set)
    for node in range(adj.shape[0]):
        print("{}/{}".format(node + 1, adj.shape[0]))
        nbrs = adj[node, :].toarray()[0]
        mask = nbrs > 0.01
        locate = np.arange(adj.shape[0])
        nbrs = locate[mask]
        [adj_dict[node].add(nbr) for nbr in nbrs]
    graphlet_node_set = set()
    for node in range(adj.shape[0]):
        if node not in adj_dict:
            continue
        nbrs = adj_dict[node]
        for first_nbr in nbrs:
            for second_nbr in nbrs:
                if first_nbr == second_nbr:
                    continue
                print("node {} first nbr: {} second nbr: {}".format(node, first_nbr, second_nbr))
                if (first_nbr not in adj_dict or second_nbr not in adj_dict) or \
                        (first_nbr not in adj_dict[second_nbr] and second_nbr not in adj_dict[first_nbr]):
                    continue
                triangle_list = [node, first_nbr, second_nbr]
                triangle_list = sorted(triangle_list)
                graphlet_node_set.add((triangle_list[0], triangle_list[1], triangle_list[2]))
    print(graphlet_node_set)
    print(len(graphlet_node_set))
    with open("graphlet_node_set.pkl", "wb") as f:
        pickle.dump(graphlet_node_set, f)
    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], \
           data['test_index']


if __name__ == '__main__':
    statistics_data("../data/")
