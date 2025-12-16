# We converted DINO-Foresight into a spatio-temporal DiT. Minimal changes done.

import math

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
  def __init__(self, dim, hidden_dim, dropout=0., out_dim=None, ):
    """ Initialize the Multi-Layer Perceptron (MLP).
        :param:
            dim        -> int : Dimension of the input
            dim        -> int : Dimension of the hidden layer
            dim        -> float : Dropout rate
    """
    super().__init__()
    if out_dim is None:
      out_dim = dim

    self.net = nn.Sequential(
        nn.Linear(dim, hidden_dim, bias=True),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim, bias=True),
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


def modulate(x, shift, scale):
  return x * (1 + scale) + shift


class Attention(nn.Module):
  def __init__(
      self,
      embed_dim,
      num_heads,
      dropout=0.,
      norm=nn.RMSNorm,
      use_qk_norm=True,
      causal=False
  ):
    super(Attention, self).__init__()

    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads
    self.dropout = dropout

    self.qkv_attn = nn.Linear(embed_dim, embed_dim * 3)
    self.project = nn.Linear(embed_dim, embed_dim)

    self.use_qk_norm = use_qk_norm
    self.causal = causal

    if self.use_qk_norm:
      self.q_norm = norm(self.head_dim)
      self.k_norm = norm(self.head_dim)

  def forward(self, x: torch.Tensor):
    B, N, C = x.size()

    qkv = self.qkv_attn(x)
    q, k, v = qkv.split(self.embed_dim, dim=2)

    q = q.view(B, N, self.num_heads, self.head_dim)
    k = k.view(B, N, self.num_heads, self.head_dim)

    if self.use_qk_norm:
      q = self.q_norm(q)
      k = self.k_norm(k)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    y = F.scaled_dot_product_attention(
        q, k, v,
        dropout_p=self.dropout,
        is_causal=self.causal
    )
    y = y.transpose(1, 2).contiguous().view(B, N, C)
    y = self.project(y)

    return y, None


class AdaptivePreNorm(nn.Module):
  def __init__(self, dim, fn):
    super().__init__()
    self.norm = nn.LayerNorm(
        dim,
        elementwise_affine=False
    )
    self.fn = fn

  def forward(self, x, scale, shift, gate, **kwargs):
    out = self.fn(
        modulate(self.norm(x), scale, shift),
        **kwargs
    )
    if isinstance(self.fn, Attention):
      f, _ = out
    else:
      f = out
    return gate * f


class TransformerEncoderSeparableAttention(nn.Module):
  def __init__(
      self,
      dim,
      depth,
      heads,
      mlp_dim,
      dropout=0.,
      window_size=1,
  ):
    super().__init__()
    self.window_size = window_size
    self.layers = nn.ModuleList([])
    self.scale_shift = nn.Parameter(torch.randn(9, dim) / dim**0.5)
    for _ in range(depth):
      self.layers.append(nn.ModuleList([
          AdaptivePreNorm(
              dim,
              Attention(
                  dim,
                  heads,
                  dropout=dropout,
                  use_qk_norm=True,
              )
          ),  # temporal
          AdaptivePreNorm(
              dim,
              Attention(
                  dim,
                  heads,
                  dropout=dropout,
                  use_qk_norm=True,
              )
          ),  # spatial
          AdaptivePreNorm(
              dim,
              FeedForward(dim, mlp_dim, dropout=dropout)
          )
      ]))

  def forward(self, x, time, full_shape):
    b, t, h, w, d = full_shape
    # time: [b, 9*d] or [b, 9, d]
    time = time.reshape(b, 9, -1)
    (
        temp_shift_msa,
        temp_scale_msa,
        temp_gate_msa,
        spatial_shift_msa,
        spatial_scale_msa,
        spatial_gate_msa,
        shift_mlp,
        scale_mlp,
        gate_mlp
    ) = (self.scale_shift[None] + time).chunk(9, dim=1)

    for attn_temporal, attn_spatial, ff in self.layers:
      # ===== Temporal attention =====
      if self.window_size == 1:
        x = einops.rearrange(
            x,
            'b (t h w) c -> (b h w) t c',
            b=b,
            t=t,
            h=h,
            w=w
        )
        # expand conds to (b*h*w, 1, d)
        reps = h * w
      else:
        x = einops.rearrange(
            x,
            'b (t h w) c -> b t h w c',
            b=b,
            t=t,
            h=h,
            w=w)
        x = einops.rearrange(
            x,
            'b t (h k1) (w k2) c -> (b h w) (t k1 k2) c',
            k1=self.window_size,
            k2=self.window_size
        )
        reps = (h // self.window_size) * (w // self.window_size)
      temp_scale = einops.repeat(
          temp_scale_msa,
          'b 1 d -> (b r) 1 d',
          r=reps
      )
      temp_shift = einops.repeat(
          temp_shift_msa,
          'b 1 d -> (b r) 1 d',
          r=reps
      )
      temp_gate = einops.repeat(
          temp_gate_msa,
          'b 1 d -> (b r) 1 d',
          r=reps
      )
      attn_val = attn_temporal(
          x,
          temp_scale,
          temp_shift,
          temp_gate
      )
      x = x + attn_val
      # ===== Spatial attention =====
      if self.window_size == 1:
        x = einops.rearrange(
            x, '(b h w) t c -> (b t) (h w) c',
            b=b,
            t=t,
            h=h,
            w=w
        )
      else:
        x = einops.rearrange(
            x,
            '(b hw) (tk) c -> b tk hw c',
            b=b,
            hw=reps
        )
        x = einops.rearrange(
            x,
            'b (t k1 k2) (h w) c -> (b t) (h k1 w k2) c',
            t=t,
            h=h // self.window_size,
            w=w // self.window_size,
            k1=self.window_size,
            k2=self.window_size
        )
      spatial_scale = einops.repeat(
          spatial_scale_msa,
          'b 1 d -> (b t) 1 d',
          t=t
      )
      spatial_shift = einops.repeat(
          spatial_shift_msa,
          'b 1 d -> (b t) 1 d',
          t=t
      )
      spatial_gate = einops.repeat(
          spatial_gate_msa,
          'b 1 d -> (b t) 1 d',
          t=t
      )
      attn_val = attn_spatial(
          x,
          spatial_scale,
          spatial_shift,
          spatial_gate
      )
      x = x + attn_val
      # ===== FFN block =====
      x = einops.rearrange(
          x,
          '(b t) (hw) c -> b (t hw) c',
          b=b,
          t=t,
          hw=h * w
      )
      # FFN conds can stay [b,1,d] and broadcast over tokens
      x = x + ff(
          x,
          shift_mlp,
          scale_mlp,
          gate_mlp
      )

    return x, None


class TimestepEmbedder(nn.Module):
  """
  Embeds scalar timesteps into vector representations.
  """

  def __init__(self, hidden_size, frequency_embedding_size=256):
    super().__init__()
    self.mlp = FeedForward(
        frequency_embedding_size,
        4 * hidden_size,
        out_dim=hidden_size
    )
    # Initialize timestep embedding MLP:
    nn.init.normal_(self.mlp.net[0].weight, std=0.02)
    nn.init.normal_(self.mlp.net[3].weight, std=0.02)
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0,
                                                           end=half, dtype=torch.float32) / half).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat(
          [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  def forward(self, t):
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_emb = self.mlp(t_freq)
    return t_emb


class FinalLayer(nn.Module):
  """
  The final layer of DiT.
  """

  def __init__(self, hidden_size, out_channels):
    super().__init__()
    self.norm_final = nn.LayerNorm(
        hidden_size,
        elementwise_affine=False,
        eps=1e-6
    )
    self.linear = nn.Linear(
        hidden_size,
        out_channels,
        bias=True
    )
    self.adaLN_modulation = nn.Sequential(
        nn.GELU(),
        nn.Linear(hidden_size, 2 * hidden_size, bias=True)
    )

  def forward(self, x, c):
    shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
    x = modulate(
        self.norm_final(x),
        shift.unsqueeze(1),
        scale.unsqueeze(1)
    )
    x = self.linear(x)
    return x


class MaskTransformer(nn.Module):
  def __init__(
      self,
      shape,
      embedding_dim=768,
      hidden_dim=768,
      depth=24,
      heads=8,
      mlp_dim=3072,
      dropout=0.1,
      use_fc_bias=False,
      separable_attention=True,
      separable_window_size=1,
  ):
    """ Initialize the Transformer model.
        :param:
            #img_size       -> int:     Input image size (default: 256)
            hidden_dim     -> int:     Hidden dimension for the transformer (default: 768)
            codebook_size  -> int:     Size of the codebook (default: 1024)
            depth          -> int:     Depth of the transformer (default: 24)
            heads          -> int:     Number of attention heads (default: 8)
            mlp_dim        -> int:     MLP dimension (default: 3072)
            dropout        -> float:   Dropout rate (default: 0.1)
            # nclass         -> int:     Number of classes (default: 1000)
    """
    super().__init__()
    self.pos_embd = AddBroadcastPosEmbed(shape=shape, embd_dim=hidden_dim)
    self.time_embd = TimestepEmbedder(hidden_size=hidden_dim)
    self.modulation = FeedForward(
        hidden_dim,
        4 * hidden_dim,
        out_dim=9 * hidden_dim
    )

    self.seperable_attention = separable_attention

    if separable_attention:
      self.transformer = TransformerEncoderSeparableAttention(
          dim=hidden_dim,
          depth=depth,
          heads=heads,
          mlp_dim=mlp_dim,
          dropout=dropout,
          window_size=separable_window_size,
      )
    else:
      raise NotImplementedError()

    self.fc_in = nn.Linear(embedding_dim, hidden_dim, bias=use_fc_bias)
    self.fc_in.weight.data.normal_(std=0.02)

    nn.init.constant_(self.modulation.net[3].weight, 0)
    nn.init.constant_(self.modulation.net[3].bias, 0)

    # this one should be modulated
    self.fc_out = FinalLayer(hidden_dim, embedding_dim)
    nn.init.constant_(self.fc_out.linear.weight, 0)
    nn.init.constant_(self.fc_out.linear.bias, 0)
    nn.init.constant_(self.fc_out.adaLN_modulation[-1].weight, 0)
    nn.init.constant_(self.fc_out.adaLN_modulation[-1].bias, 0)

  def forward(self, x, time, y=None, drop_label=None, return_attn=False):
    """ Forward.
        :param:
            x              -> torch.Tensor: bsize x t x 16 x 16 x d, the encoded tokens
            time           -> torch.Tensor: bsize
            y              -> torch.LongTensor: condition class to generate
            drop_label     -> torch.BoolTensor: either or not to drop the condition
            return_attn    -> Bool: return the attn for visualization
        :return:
            logit:         -> torch.FloatTensor: bsize x path_size*path_size * 1024, the predicted logit
            attn:          -> list(torch.FloatTensor): list of attention for visualization
    """
    b, t, h, w, c = x.size()

    # Position embedding
    x = self.fc_in(x)
    x = self.pos_embd(x)
    x = einops.rearrange(x, 'b t h w c -> b (t h w) c')

    # Temporal embedding
    time = self.time_embd(time)
    mod = self.modulation(time)

    # transformer forward pass
    x, attn = self.transformer(x, mod, full_shape=[b, t, h, w, c])
    x_out = self.fc_out(x, time)
    if return_attn:
      return x_out, attn
    else:
      return x_out


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
    b, t, h, w, c = x.shape
    shape = x.shape[1:-1]
    embs = []

    for i in range(self.n_dim):
      e = self.emb[f'd_{i}']
      if i == 0:
        e = e[:t, ...]

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
