import dgl
import torch
import torch.nn as nn
import numpy as np
import sklearn.metrics as m
from .preprocessing import kfold

def compute_ranking_metrics(true, pred, edges, neg_sampling=100):
  n = true.shape[0]
  # add self loops
  true = true + torch.eye(n)

  hits1 = 0
  hits3 = 0
  hits10 = 0
  mr = 0
  mrr = 0
  for pos in edges:
    scores = [pred[tuple(pos)]]
    i = np.random.choice([0,1])
    for j in np.random.choice(np.arange(n)[true[i] == 0], neg_sampling, False):
      neg = pos.copy()
      neg[(i-1)%2] = j
      scores.append(pred[tuple(neg)])
    rank = float(np.argsort(scores)[::-1].argmin() + 1)
    hits10 += 1 if rank <= 10 else 0
    hits3 += 1 if rank <= 3 else 0
    hits1 += 1 if rank == 1 else 0
    mr += rank
    mrr += 1 / rank

  metrics = {'mrr': mrr / len(edges),
             'mr': mr / len(edges),
             'hits1': hits1 / len(edges),
             'hits3': hits3 / len(edges),
             'hits10': hits10 / len(edges)}
  return metrics

def compute_metrics(pos_true, neg_true, preds):
  n = pos_true.shape[0]
  pos_pred = []
  neg_pred = []
  for i in range(n):
    pos_pred.append(preds[tuple(pos_true[i])])
    neg_pred.append(preds[tuple(neg_true[i])])
  p = np.hstack((np.asarray(pos_pred), np.asarray(neg_pred)))
  t = np.hstack((np.ones(n), np.zeros(n)))

  metrics = {'loss': float(m.log_loss(t, p)),
             'ap': float(m.average_precision_score(t, p)),
             'auc': float(m.roc_auc_score(t, p))}
  return metrics

def train_kfold(
    model,            # Model to train
    A,                # Graph adjacency matrix
    X,                # Node's features
    k=10,             # K fold
    lr=1e-2,          # Learning rate
    wd=1e-5,          # Weight decay
    valid_size=0.05,  # Validation set size
    epochs_max=300,   # Max number of epochs
    patience_max=30,  # Max number of epochs to wait before early stopping (if no improvements)
    dgl_model=True,   # Must be True if the model is implemented via DGL library, False otherwise
    device='cpu'):
  """
  Train a specific model using k-fold cross validation, returns a list of metrics
  computed on each fold
  """
  fold = 1
  metrics = []
  for A_train, train_e, test_e, test_ef, val_e, val_ef in kfold(A, k, valid_size):
    # reset weights
    model.reset_parameters()
    if dgl_model:
      A_train_dgl = dgl.from_scipy(A_train)
      A_train_dgl = A_train_dgl.add_self_loop()

    # scipy sparse matrix -> torch dense tensor
    n = A_train.sum()
    A_train = torch.from_numpy(A_train.todense()).to(device)
    pos_w = (A_train.shape[0] * A_train.shape[0] - n) / n
    norm = A_train.shape[0] * A_train.shape[0] / ((A_train.shape[0] * A_train.shape[0] - n) * 2)
    A_train_w = A_train.clone().flatten()
    A_train_w[A_train_w == 1] = pos_w
    A_train_w[A_train_w == 0] = 1

    model.train()
    losses = {'train_loss': [], 'valid_loss': []}
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss = nn.BCELoss(weight=A_train_w)
    patience_act = 0
    min_loss = np.inf
    for epoch in range(epochs_max):
      if patience_act >= patience_max:
        print(f'[fold {fold}] max patience reached after {patience_max} epochs, training stopped')
        break
      model.zero_grad()
      out = model(A_train_dgl if dgl_model else A_train, X)
      l = norm * loss(out.flatten(), A_train.flatten())
      l.backward()
      opt.step()
      val_metrics = compute_metrics(val_e, val_ef, out.detach().cpu())
      if val_metrics['loss'] < min_loss:
        min_loss = val_metrics['loss']
        model_state = model.state_dict()
        patience_act = 0
      elif epoch > 50:
        patience_act += 1
      if epoch % 10 == 0:
        losses['train_loss'].append(np.float16(l.item()))
        losses['valid_loss'].append(np.float16(val_metrics['loss']))
      print(f'[fold {fold}] epoch {epoch}: train loss = {l:.4f}, valid loss = {val_metrics["loss"]:.4f}, valid AUC = {val_metrics["auc"]:.4f}')

    # Load best model based on validation set metrics
    model.eval()
    model.load_state_dict(model_state)
    with torch.no_grad():
      out = model(A_train_dgl if dgl_model else A_train, X).cpu()
      test_metrics = compute_metrics(test_e, test_ef, out)
      test_ranking = compute_ranking_metrics(torch.Tensor(A.todense()), out, test_e)
    print(f'[fold {fold}] test metrics: {test_metrics}')
    print(f'[fold {fold}] test mrr: {test_ranking}')
    test_metrics.update(test_ranking)
    test_metrics.update(losses)
    metrics.append(test_metrics)
    fold += 1
  return metrics
