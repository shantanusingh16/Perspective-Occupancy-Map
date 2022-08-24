import math
import torch
import torch.nn as nn

from einops import rearrange


#################################### Positional Encoding ######################################
'''
SRC url: https://github.com/BenSaunders27/ProgressiveTransformersSLP/blob/adbd3e9ea9f1b20063d84021a0d6eb9a124ebb87/transformer_layers.py#L98
'''
class PositionalEncoding(nn.Module):

    def __init__(self, size: int = 0, max_len: int = 200000, mask_count=False):

        if size % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(size))
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, size, 2, dtype=torch.float) *
                              -(math.log(10000.0) / size)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, size]

        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dim = size
        self.mask_count = mask_count

    def forward(self, emb):

        return emb + self.pe[:, :emb.size(1)]


#################################### FeedForward Block ########################################

class FeedForward1D(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., skip_conn=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.skip_conn = skip_conn

    def forward(self, x):
        out = self.net(x)
        if self.skip_conn:
            out = out + x
        return out


class FeedForward2D(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., skip_conn=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.skip_conn = skip_conn

    def forward(self, x):
        B, C, H, W = x.shape
        x =  x.reshape((B, C, -1))
        out = self.net(x)
        if self.skip_conn:
            out = out + x
        return out.reshape((B, C, H, W))



#################################### MHA Block ########################################

class MHA1D(nn.Module):
    def __init__(self, in_dim, heads, dim_head, dropout=0, skip_conn=True):
        super(MHA1D, self).__init__()

        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == in_dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(in_dim, inner_dim, bias = False)
        self.to_k = nn.Linear(in_dim, inner_dim, bias = False)
        self.to_v = nn.Linear(in_dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, in_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        self.scores = None
        self.skip_conn = skip_conn


    def forward(self, x_key, x_query=None, x_value=None):
        if x_query is None :
            x_query = x_key.clone()
        
        if x_value is None :
            x_value = x_key.clone()

        # x_key.shape = x_value.shape = B, M, D
        # x_query.shape = B, N, D

        if self.skip_conn:
            assert x_query.size(1) == x_value.size(1)

        k = self.to_k(x_key)  # B, M, H*d
        q = self.to_q(x_query)  # B, N, H*d
        v = self.to_v(x_value)  # B, M, H*d

        k, q, v = map(lambda x : rearrange(x, 'b n (h d) -> b h n d', h=self.heads), [k, q, v])  # B H (M, N, M) d

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # B H N M

        attn = self.attend(dots)
        
        self.scores = attn

        T = torch.matmul(attn, v)   # B H N d
        out = rearrange(T, 'b h n d -> b n (h d)')  # B N H*d
        out = self.to_out(out)   # B N D

        if self.skip_conn:
            out += x_value

        return out


class MHA2D(nn.Module):
    def __init__(self, in_dim, heads, dim_head, dropout=0, skip_conn=True):
        super(MHA2D, self).__init__()

        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == in_dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(in_dim, inner_dim, bias = False)
        self.to_k = nn.Linear(in_dim, inner_dim, bias = False)
        self.to_v = nn.Linear(in_dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, in_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        self.scores = None
        self.skip_conn = skip_conn


    def forward(self, x_key, x_query=None, x_value=None):
        if x_query is None :
            x_query = x_key.clone()
        
        if x_value is None :
            x_value = x_key.clone()

        if self.skip_conn:
            assert x_query.size(1) == x_value.size(1)

        B, C, H, W = x_value.shape

        x_key = x_key.reshape((*x_key.shape[:2], -1)).transpose(-1,-2)  # Convert B x C x H X W -> B x HW x C
        x_query = x_query.reshape((*x_query.shape[:2], -1)).transpose(-1,-2)  # Convert B x C x H X W -> B x HW x C
        x_value = x_value.reshape((*x_value.shape[:2], -1)).transpose(-1,-2)  # Convert B x C x H X W -> B x HW x C

        k = self.to_k(x_key)
        q = self.to_q(x_query)
        v = self.to_v(x_value)

        k, q, v = map(lambda x : rearrange(x, 'b n (h d) -> b h n d', h=self.heads), [k, q, v])

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        
        self.scores = attn

        T = torch.matmul(attn, v)
        out = rearrange(T, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        if self.skip_conn:
            out += x_value
        out = out.reshape((B, C, H, W))
        return out


#################################### Transformer Block ########################################

class Transformer1D(nn.Module):
    def __init__(self, in_dim, nheads, head_dim, mha_skipcon=True, dropout=0, ff_dim=64, ff_skipcon=True):
        super(Transformer1D, self).__init__()
        self.mha_skipcon = mha_skipcon
        self.mha = MHA1D(in_dim, nheads, head_dim, dropout=dropout, skip_conn=mha_skipcon)
        self.ffd = FeedForward1D(in_dim, ff_dim, skip_conn=ff_skipcon, dropout=dropout)


    def forward(self, x_key, x_query=None, x_value=None):
        y_hat = self.mha(x_key, x_query, x_value)
        return self.ffd(y_hat)


class Transformer2D(Transformer1D):
    def __init__(self, in_dim, nheads, head_dim, mha_skipcon=True, dropout=0, ff_dim=64, ff_skipcon=True):
        super(Transformer2D, self).__init__(in_dim, nheads, head_dim, mha_skipcon=mha_skipcon, dropout=dropout, ff_dim=ff_dim, ff_skipcon=ff_skipcon)

    def forward(self, x_key, x_query=None, x_value=None):
        if x_value is not None:
            B, C, H, W = x_value.shape
        else:
            B, C, H, W = x_key.shape

        x_key = x_key.reshape((*x_key.shape[:2], -1)).transpose(-1,-2)  # Convert B x C x H X W -> B x HW x C

        if x_query is not None:
            x_query = x_query.reshape((*x_query.shape[:2], -1)).transpose(-1,-2)  # Convert B x C x H X W -> B x HW x C
        
        if x_value is not None:
            x_value = x_value.reshape((*x_value.shape[:2], -1)).transpose(-1,-2)  # Convert B x C x H X W -> B x HW x C

        y_hat = self.mha(x_key, x_query, x_value)
        y_hat = self.ffd(y_hat)

        return y_hat.reshape(B, C, H, W)


if __name__ == '__main__':
    a = PositionalEncoding(32, 1024)
    # features = torch.arange(0, 24)
    features = torch.arange(0, 65536)
    features = torch.where(features < 20, features, torch.zeros_like(features))
    # features = features.view([2, 3, 4]).float()
    features = features.view([8, 128, 8, 8]).float()

    features2 = torch.arange(0, 65536)
    features2 = torch.where(features2 < 20, features2, torch.zeros_like(features2))
    # features = features.view([2, 3, 4]).float()
    features2 = features2.view([8, 128, 8, 8]).float()

    features3 = torch.arange(0, 65536)
    features3 = torch.where(features3 < 20, features3, torch.zeros_like(features3))
    # features = features.view([2, 3, 4]).float()
    features3 = features3.view([8, 128, 8, 8]).float()

    attention3 = MHA2D(128)
    print(attention3(features, features2, features3).shape)