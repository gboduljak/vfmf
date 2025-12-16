# BERT architecture for the Masked Bidirectional Encoder Transformer
import einops
import torch
import torch.nn.functional as F
from torch import nn


class PreNorm(nn.Module):

  def __init__(self, dim, fn):
    """ PreNorm module to apply layer normalization before a given function
        :param:
            dim  -> int: Dimension of the input
            fn   -> nn.Module: The function to apply after layer normalization
        """
    super().__init__()
    self.norm = nn.LayerNorm(dim)
    self.fn = fn

  def forward(self, x, **kwargs):
    """ Forward pass through the PreNorm module
        :param:
            x        -> torch.Tensor: Input tensor
            **kwargs -> _ : Additional keyword arguments for the function
        :return
            torch.Tensor: Output of the function applied after layer normalization
    """
    return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
  def __init__(self, dim, hidden_dim, dropout=0.):
    """ Initialize the Multi-Layer Perceptron (MLP).
        :param:
            dim        -> int : Dimension of the input
            dim        -> int : Dimension of the hidden layer
            dim        -> float : Dropout rate
    """
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(dim, hidden_dim, bias=True),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim, bias=True),
        nn.Dropout(dropout)
    )

  def forward(self, x):
    """ Forward pass through the MLP module.
        :param:
            x -> torch.Tensor: Input tensor
        :return
            torch.Tensor: Output of the function applied after layer
    """
    return self.net(x)


class Attention(nn.Module):
  def __init__(
          self,
          embed_dim,
          num_heads,
          dropout=0.,
          norm=nn.RMSNorm,
          use_qk_norm=True):
    super(Attention, self).__init__()

    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads
    self.dropout = dropout

    self.qkv = nn.Linear(embed_dim, embed_dim * 3)
    self.project = nn.Linear(embed_dim, embed_dim)

    self.use_qk_norm = use_qk_norm

    if self.use_qk_norm:
      self.q_norm = norm(self.head_dim)
      self.k_norm = norm(self.head_dim)

  def forward(self, x: torch.Tensor):
    B, N, C = x.size()

    qkv = self.qkv(x)
    q, k, v = qkv.split(self.embed_dim, dim=2)

    q = q.view(B, N, self.num_heads, self.head_dim)
    k = k.view(B, N, self.num_heads, self.head_dim)

    if self.use_qk_norm:
      q = self.q_norm(q)
      k = self.k_norm(k)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
    y = y.transpose(1, 2).contiguous().view(B, N, C)
    y = self.project(y)

    return y


class Transformer(nn.Module):
  def __init__(
      self,
      dim,
      depth,
      heads,
      mlp_dim,
      dropout=0.,
      use_qk_norm=True,
      num_registers=0
  ):
    """ Initialize the Attention module.
        :param:
            dim       -> int : number of hidden dimension of attention
            depth     -> int : number of layer for the transformer
            heads     -> int : Number of heads
            mlp_dim   -> int : number of hidden dimension for mlp
            dropout   -> float : Dropout rate
    """
    super().__init__()
    if num_registers:
      self.num_registers = num_registers
      self.register_tokens = (
          nn.Parameter(torch.zeros(1, num_registers, dim))
      )
      nn.init.normal_(self.register_tokens, std=1e-6)
    self.layers = nn.ModuleList([])
    for _ in range(depth):
      self.layers.append(nn.ModuleList([
          PreNorm(
              dim,
              Attention(
                  dim,
                  heads,
                  dropout=dropout,
                  use_qk_norm=use_qk_norm
              )
          ),
          PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
      ]))

  def forward(self, x, full_shape=None):
    """ Forward pass through the Attention module.
        :param:
            x -> torch.Tensor: Input tensor
        :return
            x -> torch.Tensor: Output of the Transformer
            l_attn -> list(torch.Tensor): list of the attention
    """
    b, *_ = x.shape
    if hasattr(self, "register_tokens"):
      x_reg = self.register_tokens.expand(b, -1, -1)
      x = torch.cat([x_reg, x], dim=1)
    for attn, ff in self.layers:
      x = attn(x) + x
      x = ff(x) + x
    if hasattr(self, "register_tokens"):
      x = x[:, self.num_registers:, :]
    return x


################ Spatiotemporal broadcasted positional embeddings ########


class AddBroadcastPosEmbed(nn.Module):
  def __init__(self, shape, embd_dim, dim=-1):
    super().__init__()
    assert dim in [-1, 1]  # only first or last dim supported
    self.shape = shape
    self.n_dim = n_dim = len(shape)
    self.embd_dim = embd_dim
    self.dim = dim

    assert embd_dim % n_dim == 0, f"{embd_dim} % {n_dim} != 0"
    self.emb = nn.ParameterDict(
        {
            f'd_{i}': nn.init.trunc_normal_(
                nn.Parameter(
                    torch.randn(
                        self.shape[i],
                        embd_dim //
                        n_dim)),
                0.,
                0.02) if dim == -
            1 else nn.init.trunc_normal_(
                torch.randn(
                    embd_dim //
                    n_dim,
                    self.shape[i]),
                0.,
                0.02) for i in range(n_dim)})
    # print("self.emb",self.emb)

  def forward(self, x, decode_step=None, decode_idx=None):
    shape = x.shape[1:-1]
    embs = []

    for i in range(self.n_dim):
      e = self.emb[f'd_{i}']

      if self.dim == -1:
        # (1, 1, ..., 1, self.shape[i], 1, ..., -1)
        e = e.view(1, *((1,) * i), shape[i],
                   *((1,) * (self.n_dim - i - 1)), -1)
        # print("e.shape",e.shape) # [1,5, 1, 1, 384] , [1,1,16,1,384],
        # [1,1,1,32,384]
        e = e.expand(1, *shape, -1)
        # print("e.expand.shape",e.shape)
      else:
        e = e.view(1, -1, *((1,) * i),
                   shape[i], *((1,) * (self.n_dim - i - 1)))
        e = e.expand(1, -1, shape)

      embs.append(e)

    embs = torch.cat(embs, dim=self.dim)
    return x + embs
