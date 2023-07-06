import torch
import numpy as np
import scipy.sparse as sp

def laplacian_eig(adj, norm=False, device='cpu'):
  if not torch.is_tensor(adj):
    adj = torch.Tensor(adj.todense())
  if norm:
    D = torch.diag(adj.sum(axis=1).clip(1).pow(-0.5))
    L = torch.eye(adj.shape[0]).to(device) - D.mm(adj.mm(D))
  else:
    D = torch.diag(adj.sum(axis=1))
    L = D - adj

  L = L.cpu()
  eig, eiv = map(lambda x: x.astype(float), np.linalg.eig(L))
  eig[eig < 1e-10] = 0
  return eig, np.asarray(eiv)

def laplacian_pe(adj, k, device='cpu'):
  eig, eiv = laplacian_eig(adj, norm=True, device=device)
  eiv = eiv[:, eig > 0]
  eig = eig[eig > 0]
  sort = np.argsort(eig)
  sign = 2 * (np.random.rand(k) > 0.5) - 1
  return torch.Tensor(sign * np.real(eiv[:, sort[:k]])).to(device)

def partition(adj, k, remove_edges=True, device='cpu'):
  eig, eiv = laplacian_eig(adj, device=device)
  nodes = eiv[:, eig[eig > 0].argmin()].flatten()

  adj_list = []
  p = np.array_split(np.argsort(nodes), k)
  for i, part in enumerate(p):
    if remove_edges:
      A = adj.copy()
      # All nodes not present in the current partition
      not_p = np.hstack(np.delete(p, i, axis=0))
      A[not_p,:] = 0
      A[:,not_p] = 0
      A.eliminate_zeros()
    else:
      A = sp.dok_matrix(adj.shape, dtype=np.float32).tocsr()
      A[part,:] = adj[part,:]
      A[:,part] = adj[:,part]
      A.eliminate_zeros()
    adj_list.append(A)
  return adj_list, p

def random_edge_splitting(adj, k):
  adj = sp.triu(adj)
  if not sp.isspmatrix_coo(adj):
    adj = adj.tocoo()
  edges_idx = np.arange(adj.sum())
  np.random.shuffle(edges_idx)
  p = np.array_split(edges_idx, k)
  adj_list = []
  for part in p:
    part = part.astype(np.int32)
    rows = np.hstack((adj.row[part], adj.col[part]))
    cols = np.hstack((adj.col[part], adj.row[part]))
    A = sp.csr_matrix((np.ones(len(part) * 2), (rows, cols)), adj.shape, dtype=np.float32)
    adj_list.append(A)
  return adj_list

