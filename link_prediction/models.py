import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import laplacian_pe

class GNCN(nn.Module):
  def __init__(self, in_features, out_features, s=1.8, activation=lambda x: x, dropout=0):
    super().__init__()
    self.drop = nn.Dropout(dropout)
    self.weights = nn.Linear(in_features, out_features, bias=False)
    self.act = activation
    self.s = s
  def reset_parameters(self):
    self.weights.reset_parameters()
  def forward(self, A, X):
    X = self.drop(X)
    X = F.normalize(self.weights(X))
    return self.act(self.s * A.mm(X))

class InputEncoder(nn.Module):
  def __init__(self, feat_dim, emb_dim, pe_enc=False, device='cpu'):
    super().__init__()
    self.linear = nn.Linear(feat_dim, emb_dim)
    self.pe_enc = pe_enc
    self.device = device
    self.init = True if self.pe_enc else False
  def reset_parameters(self):
    self.init = True if self.pe_enc else False
    for layer in self.children():
      if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()
  def forward(self, A, X):
    if self.init:
      print("Computing laplacian positional embeddings...")
      self.pe = laplacian_pe(A, self.linear.out_features, self.device)
      self.init = False
    if self.pe_enc:
      return self.linear(X) + self.pe
    else:
      return self.linear(X)

class GTAE(nn.Module):
  def __init__(self, feat_dim, emb_dim, latent_size, n_heads, n_layers=2, pe_enc=False, device='cpu'):
    super().__init__()
    self.input = InputEncoder(feat_dim, emb_dim, pe_enc, device)
    t_enc = nn.TransformerEncoderLayer(emb_dim, n_heads, emb_dim, activation=F.gelu, dropout=0.1)
    self.t_enc_stack = nn.TransformerEncoder(t_enc, n_layers)
    self.linear = nn.Linear(emb_dim, latent_size)
  def reset_parameters(self):
    for layer in self.children():
      if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()
  def forward(self, A, X):
    self.enc = self.linear(self.t_enc_stack(self.input(A, X), A))
    return F.sigmoid(self.enc.mm(self.enc.T))

class GCAE(nn.Module):
  def __init__(self, in_feat, hid_size, latent_size):
    super().__init__()
    self.conv1 = dgl.nn.GraphConv(in_feat, hid_size, activation=F.relu)
    self.conv2 = dgl.nn.GraphConv(hid_size, latent_size)
    self.proj = nn.Sequential(
        nn.Linear(latent_size, latent_size * 2, bias=False),
        nn.ReLU(),
        nn.Linear(latent_size * 2, latent_size, bias=False)
    )
  def reset_parameters(self):
    for layer in self.children():
      if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()
  def forward(self, A, X, need_proj=False):
    self.enc = self.conv2(A, self.conv1(A, X))
    if need_proj:
      return F.sigmoid(self.enc.mm(self.enc.T)), self.proj(self.enc)
    else:
      return F.sigmoid(self.enc.mm(self.enc.T))

class GATAE(nn.Module):
  def __init__(self, in_feat, hid_size, latent_size, n_heads1, n_heads2):
    super().__init__()
    if hid_size % n_heads1 != 0:
      return Exception(f"hid_size ({hid_size}) must be divisible by n_heads1 ({n_heads1})")
    self.conv1 = dgl.nn.GATConv(in_feat, int(hid_size / n_heads1), n_heads1, activation=F.elu)
    self.conv2 = dgl.nn.GATConv(hid_size, latent_size, n_heads2)
    self.proj = nn.Sequential(
        nn.Linear(latent_size, latent_size * 2, bias=False),
        nn.ReLU(),
        nn.Linear(latent_size * 2, latent_size, bias=False)
    )
  def reset_parameters(self):
    for layer in self.children():
      if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()
  def forward(self, A, X, need_proj=False):
    # In the first attention layer head's output are concatenated, in the second
    # they're averaged
    out, att = self.conv1(A, X, get_attention=True)
    self.enc = self.conv2(A, out.flatten(1)).mean(1)
    if need_proj:
      return F.sigmoid(self.enc.mm(self.enc.T)), self.proj(self.enc)
    else:
      return F.sigmoid(self.enc.mm(self.enc.T)), att

class GNCAE(nn.Module):
  def __init__(self, in_feat, hid_size, latent_size, s=1.8):
    super().__init__()
    self.conv1 = GNCN(in_feat, hid_size, s=s, activation=F.relu)
    self.conv2 = GNCN(hid_size, latent_size, s=s)
    self.proj = nn.Sequential(
        nn.Linear(latent_size, latent_size * 2, bias=False),
        nn.ReLU(),
        nn.Linear(latent_size * 2, latent_size, bias=False)
    )
  def reset_parameters(self):
    for layer in self.children():
      if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()
  def normalize(self, A):
    # Add self loops
    A += torch.eye(A.shape[0], out=torch.empty_like(A))
    # Compute degree matrix
    D = A.sum(axis=1).pow(-0.5)
    return (D[None,:].T * A * D).to_sparse()
  def forward(self, A, X, need_proj=False):
    A = self.normalize(A)
    self.enc = self.conv2(A, self.conv1(A, X))
    if need_proj:
      return F.sigmoid(self.enc.mm(self.enc.T)), self.proj(self.enc)
    else:
      return F.sigmoid(self.enc.mm(self.enc.T))

