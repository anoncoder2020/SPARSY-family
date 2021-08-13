import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import copy
import math
import os

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def save_sparse_csr(filename,array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix(
        (loader['data'], loader['indices'], loader['indptr']),
        shape=loader['shape']
    )

def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended
    if 'nell.0' in dataset_str:
        # Find relation nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(allx.shape[0], len(graph))
        isolated_node_idx = np.setdiff1d(test_idx_range_full, test_idx_reorder)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - allx.shape[0], :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - allx.shape[0], :] = ty
        ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]

        idx_all = np.setdiff1d(range(len(graph)), isolated_node_idx)

        if not os.path.isfile("data/{}.features.npz".format(dataset_str)):
            print("Creating feature vectors for relations - this might take a while...")
            features_extended = sp.hstack(
                (features, sp.lil_matrix(
                    (features.shape[0], len(isolated_node_idx))
                )), dtype=np.float32).todense()
            features_extended[isolated_node_idx, features.shape[1]:] = np.eye(
                len(isolated_node_idx), dtype=np.float32)
            features = sp.csr_matrix(features_extended)
            print("Done!")
            save_sparse_csr("data/{}.features".format(dataset_str),
                            features)
        else:
            features = load_sparse_csr(
                "data/{}.features.npz".format(dataset_str))

        whole_adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    else:
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        whole_adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))


    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    return features, graph, y, test_idx_range, labels, whole_adj



def build_cluster_for_object(chosen_cluster, clusters):
    idx_cluster1 = np.zeros(5)
    for i in range(len(chosen_cluster)):
        if i == 0:
            idx_cluster1 = np.where(clusters == chosen_cluster[i])[0]
        else:
            idx_cluster2 = np.where(clusters == chosen_cluster[i])[0]
            idx_cluster1 = np.concatenate((idx_cluster1, idx_cluster2))
    return idx_cluster1



def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1), dtype=float)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def remove_in_cycle(adj, percentage):
    whole_adj = copy.deepcopy(adj)
    sub_graph = nx.from_scipy_sparse_matrix(whole_adj)
    num_edges = nx.number_of_edges(sub_graph)
    cut_num_edges = math.floor(num_edges * percentage)
    total_cut = 0
    while total_cut < cut_num_edges:
        try:
            cycle_edge_list = nx.find_cycle(sub_graph, orientation='ignore')
        except Exception as NetworkXNoCycle:
            break
        idx = np.arange(len(cycle_edge_list))
        remove_edges = np.random.choice(idx, 1, replace=False)
        sub_graph.remove_edge(cycle_edge_list[remove_edges[0]][0], cycle_edge_list[remove_edges[0]][1])
        total_cut += 1
    update_adj = nx.adjacency_matrix(sub_graph)

    return update_adj, total_cut


def batch_cycle(y, test_idx_range, labels, chosen_cluster, clusters, adj, features, percent_edges, percent_trained_node):
    node_idx = build_cluster_for_object(chosen_cluster, clusters)
    node_idx = sorted(node_idx)
    node_ind_dict = dict((k, i) for i, k in enumerate(node_idx))

    s_features = features[node_idx]
    p_features = preprocess_features(s_features)

    p_adj = adj[node_idx, :][:, node_idx]

    if percent_edges > 0:
        p_adj, num_remove = remove_in_cycle(p_adj, percent_edges)
    p_support = [preprocess_adj(p_adj)]

    labels = labels[node_idx]

    idx_cluster = node_idx
    available_nodes = list(set(idx_cluster).difference(set(test_idx_range)))
    equal_idx_test = np.in1d(test_idx_range, idx_cluster)
    test_idx_in_cluster = np.where(equal_idx_test == True)
    test_idx_range = test_idx_range[test_idx_in_cluster[0]]

    diff = int(len(node_idx) * percent_trained_node - len(y))
    idx_test = test_idx_range
    idx_train = available_nodes[0:len(y) + diff]
    idx_val = available_nodes[len(y) + diff:len(y) + diff + 500]


    train_inter = sorted(list(set(idx_train).intersection(set(node_idx))))
    train_indices_label = [node_ind_dict[x] for x in train_inter]
    test_inter = sorted(list(set(idx_test).intersection(set(node_idx))))
    test_indices_label = [node_ind_dict[x] for x in test_inter]
    val_inter = sorted(list(set(idx_val).intersection(set(node_idx))))
    val_indices_label = [node_ind_dict[x] for x in val_inter]


    train_mask = sample_mask(train_indices_label, labels.shape[0])
    val_mask = sample_mask(val_indices_label, labels.shape[0])
    test_mask = sample_mask(test_indices_label, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return p_support, p_features, y_train, y_val, y_test, train_mask, val_mask, test_mask, chosen_cluster, idx_train, idx_val, idx_test


def minbatch_cycle(labels, y, test_idx_range, block_size, num_clusters, set_seed, clusters, adj, features, percent_edges, percent_trained_node):
    whole_dict = {}
    cluster_arr_id = np.arange(0, num_clusters)
    np.random.seed(seed=set_seed)
    np.random.shuffle(cluster_arr_id)
    chosen_id = []

    for idx in range(0, num_clusters, block_size):
        idx_chosen_cluster = np.arange(idx, min(idx + block_size, num_clusters))
        cluster_idx = cluster_arr_id[idx_chosen_cluster]


        p_support, p_features, y_train, y_val, y_test, train_mask, val_mask, test_mask, \
        chosen_cluster, idx_train, idx_val, idx_test = batch_cycle(y, test_idx_range, labels, cluster_idx,
                                        clusters, adj, features, percent_edges, percent_trained_node)
        whole_dict[tuple(chosen_cluster)] = (p_support, p_features, y_train, y_val, y_test, train_mask, val_mask, test_mask, idx_train, idx_val, idx_test)
        chosen_id.append(cluster_idx)
    return whole_dict, chosen_id
