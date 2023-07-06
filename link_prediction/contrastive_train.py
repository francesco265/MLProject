import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .preprocessing import kfold
from .utils import partition, random_edge_splitting
from .train import compute_metrics, compute_ranking_metrics
from itertools import combinations
from math import comb

def contrastive_loss(x_base, x_aug, temp=1.0, device='cpu'):
  proj_size = x_base.shape[0]
  mask = (torch.diag(torch.full((proj_size,), -np.inf), proj_size) + torch.diag(torch.full((proj_size,), -np.inf), -proj_size)).to(device)
  boh1 = torch.vstack((x_aug, x_base))
  boh2 = torch.vstack((x_base, x_aug))
  sim = F.normalize(boh1).mm(F.normalize(boh2).T)
  sim /= temp
  return -F.log_softmax(sim + mask, dim=1).diag().mean()

def train_contrastive_kfold(
    model,                  # Model to train
    A,                      # Graph adjacency matrix
    X,                      # Node's features
    part_type='clust_conj', # Type of partitioning to use
    n_part=2,               # Number of partitions
    contrastive_w=0.5,      # Weight coefficient given to the contrastive loss
    contrastive_t=0.5,      # Temperature used in contrastive loss softmax
    proj_size=64,           # Dimension of the node's projections, used only if part_type == 'clust_dis'
    k=10,                   # K fold
    lr=1e-2,                # Learning rate
    wd=1e-5,                # Weight decay
    valid_size=0.05,        # Validation set size
    epochs_max=300,         # Max number of epochs
    patience_max=30,        # Max number of epochs to wait before early stopping (if no improvements are made)
    dgl_model=True,         # Must be True if the model is implemented via DGL library, False otherwise
    device='cpu'):
  if part_type not in ['res', 'clust_dis', 'clust_conj']:
    return Exception('invalid partitioning type')

  fold = 1
  metrics = []
  for A_train, train_e, test_e, test_ef, val_e, val_ef in kfold(A, k, valid_size):
    # reset weights
    model.reset_parameters()

    # static partitioning
    if part_type == 'clust_dis':
      subA, subA_nodes = partition(A_train, n_part)
    elif part_type == 'clust_conj':
      subA, subA_nodes = partition(A_train, n_part, remove_edges=False)
    else:
      subA = random_edge_splitting(A_train, n_part)

    subA_dgl = []
    if dgl_model:
      for i in subA:
        tmp = dgl.from_scipy(i).to(device)
        tmp = tmp.add_self_loop()
        subA_dgl.append(tmp)

    norm = []
    loss = []
    for i in range(len(subA)):
      subA[i] = torch.from_numpy(subA[i].todense()).to(device)
      n = subA[i].sum()
      pos_w = (subA[i].shape[0] * subA[i].shape[0] - n) / n
      norm.append(subA[i].shape[0] * subA[i].shape[0] / ((subA[i].shape[0] * subA[i].shape[0] - n) * 2))
      tmp = subA[i].clone().flatten()
      tmp[tmp == 1] = pos_w
      tmp[tmp == 0] = 1
      loss.append(nn.BCELoss(weight=tmp))

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    patience_act = 0
    min_loss = np.inf
    contrastive_active = False
    for epoch in range(epochs_max):
      if patience_act >= patience_max:
        print(f'[fold {fold}] max patience reached after {patience_max} epochs, training stopped')
        break
      if epoch > 50:
        contrastive_active = True
      model.zero_grad()
      embeddings = []
      l = 0
      if part_type == 'clust_dis':
        proj_base = torch.empty((A.shape[0], proj_size))
        proj_aug = torch.empty((A.shape[0], proj_size))
      else:
        projs = []

      for i in range(len(subA)):
        out, proj = model(subA_dgl[i] if dgl_model else subA[i], X, need_proj=True)
        embeddings.append(model.enc)
        l += norm[i] * loss[i](out.flatten(), subA[i].flatten())

        if part_type == 'clust_dis':
          subA_not = np.hstack(np.delete(subA_nodes, i, axis=0))
          ## Augmented projections
          proj_aug[subA_nodes[i],:] = proj[subA_nodes[i],:]
          ## Base projections
          proj_base[subA_not,:] = proj[subA_not[i],:]
        else:
          projs.append(proj)
      l /= n_part

      # contrastive loss
      if contrastive_active:
        if part_type == 'clust_dis':
          lc = contrastive_loss(proj_base, proj_aug, contrastive_t, device=device)
        else:
          lc = 0
          for c in combinations(projs, 2):
            lc += contrastive_loss(*c, contrastive_t, device=device)
          lc /= comb(n_part, 2)
        l += contrastive_w * lc

      l.backward()
      opt.step()

      embeddings = torch.stack(embeddings).mean(dim=0)
      out = F.sigmoid(embeddings.mm(embeddings.T))
      val_metrics = compute_metrics(val_e, val_ef, out.detach().cpu())
      if val_metrics['loss'] < min_loss:
        min_loss = val_metrics['loss']
        model_state = model.state_dict()
        patience_act = 0
      elif epoch > 50:
        patience_act += 1
      print(f'[fold {fold}] epoch {epoch}: train loss = {l:.4f}, valid loss = {val_metrics["loss"]:.4f}, valid AUC = {val_metrics["auc"]:.4f}')

    # Load best model based on validation set metrics
    model.eval()
    model.load_state_dict(model_state)
    with torch.no_grad():
      embeddings = []
      for i in range(len(subA)):
        out = model(subA_dgl[i] if dgl_model else subA[i], X)
        embeddings.append(model.enc)
      embeddings = torch.stack(embeddings).mean(dim=0)
      out = F.sigmoid(embeddings.mm(embeddings.T))
      test_metrics = compute_metrics(test_e, test_ef, out.cpu())
      test_ranking = compute_ranking_metrics(torch.Tensor(A.todense()), out.cpu(), test_e)
    print(f'[fold {fold}] test metrics: {test_metrics}')
    print(f'[fold {fold}] test mrr: {test_ranking}')
    test_metrics.update(test_ranking)
    metrics.append(test_metrics)
    fold += 1
  return metrics
