import numpy as np
import scipy.sparse as sp


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    return coords

def ismember(a, b):
    return np.any(np.all((a - b) == 0, axis=-1))

# Splits data in train and test, returns an iterator to the k folds
def kfold(adj, k=10, valid=0.05):
    all_edges = sparse_to_tuple(adj)
    edges = sparse_to_tuple(sp.triu(adj))
    n_edges = edges.shape[0]
    valid_size = int(n_edges * valid)

    edges_idx = np.arange(n_edges)
    np.random.shuffle(edges_idx)
    k_train_folds = np.array_split(edges_idx, k)

    # build negative samples, one for each fold (used only for testing)
    neg_edges = np.empty((0,2), dtype=int)
    while len(neg_edges) < n_edges + valid_size * k:
      i = np.random.randint(0, adj.shape[0])
      j = np.random.randint(0, adj.shape[0])
      if ismember([i, j], all_edges):
          continue
      if ismember([i, j], neg_edges) or ismember([j, i], neg_edges):
          continue
      neg_edges = np.vstack((neg_edges, [i, j]))
    test_neg_edges = np.array_split(neg_edges[:n_edges], k)
    val_neg_edges = np.array_split(neg_edges[n_edges:], k)

    # positive samples for train, test and validation
    for i, test_edges_idx in enumerate(k_train_folds):
      train_edges_idx = np.hstack(np.delete(k_train_folds, i, axis=0))
      np.random.shuffle(train_edges_idx)

      test_edges = edges[test_edges_idx]
      val_edges = edges[train_edges_idx[:valid_size]]
      train_edges = edges[train_edges_idx[valid_size:]]

      # build train adjacency matric
      adj_train = sp.csr_matrix((np.ones(len(train_edges)), (train_edges[:,0], train_edges[:,1])), adj.shape, dtype=adj.dtype)
      adj_train = adj_train + adj_train.T
      yield adj_train, train_edges, test_edges, test_neg_edges[i], val_edges, val_neg_edges[i]

# Split data in train, test, validation
def split_data(adj, test=0.1, valid=0.05):
    all_edges = sparse_to_tuple(adj)
    edges = sparse_to_tuple(sp.triu(adj))
    # 10% test and 5% validation
    n_test = int(edges.shape[0] * test)
    n_val = int(edges.shape[0] * valid)

    edges_idx = np.arange(edges.shape[0])
    np.random.shuffle(edges_idx)
    val_edges = edges[edges_idx[:n_val]]
    test_edges = edges[edges_idx[n_val:(n_val + n_test)]]
    train_edges = edges[edges_idx[(n_val + n_test):]]

    edges_false = np.empty((0,2), dtype=int)
    while len(edges_false) < len(test_edges) + len(val_edges):
        i = np.random.randint(0, adj.shape[0])
        j = np.random.randint(0, adj.shape[0])
        if ismember([i, j], all_edges):
            continue
        if ismember([i, j], edges_false) or ismember([j, i], edges_false):
            continue
        edges_false = np.vstack((edges_false, [i, j]))
    test_edges_false, val_edges_false = edges_false[:len(test_edges)], edges_false[len(test_edges):]

    adj_train = sp.csr_matrix((np.ones(len(train_edges)), (train_edges[:,0], train_edges[:,1])), adj.shape, dtype=adj.dtype)
    adj_train = adj_train + adj_train.T

    return adj_train, train_edges, test_edges, test_edges_false, val_edges, val_edges_false
