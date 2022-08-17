import torch.nn as nn

class FeedForward(nn.Module):
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