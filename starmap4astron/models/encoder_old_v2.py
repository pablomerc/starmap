import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────────────────
# Helper: residual block with optional dilation
# ────────────────────────────────────────────────────────────────────────────
class ResBlock1D(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        padding = dilation  # keep length unchanged
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3,
                               padding=padding, dilation=dilation, bias=False)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3,
                               padding=padding, dilation=dilation, bias=False)
        self.bn2   = nn.BatchNorm1d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)          # skip connection


# ────────────────────────────────────────────────────────────────────────────
# 1‑D encoder
# ────────────────────────────────────────────────────────────────────────────
class LightcurveEncoder1D(nn.Module):
    """
    Encode a 1‑D light‑curve to a latent 2‑D grid.
    Args
    ----
    latent_channels : C   – channels in the output grid
    grid_size       : (H, W)
    in_channels     : input channels (1 for a single flux series)
    base_channels   : width of the first conv layer
    """
    def __init__(self,
                 latent_channels: int = 256,
                 grid_size: tuple[int, int] = (8, 8),
                 in_channels: int = 1,
                 base_channels: int = 64):
        super().__init__()

        self.grid_size = grid_size
        self.latent_channels = latent_channels
        H, W = grid_size
        grid_elems = H * W
        self.n_tokens = grid_elems


        # Stem: wide receptive field, stride 2 to shrink sequence length
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels,
                      kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(True)
        )
        # ── Extra down-sampling conv stages (depth ↑) ──────────────────────
        # Each stage: channels ×2, length /2
        pyramid = []
        c = base_channels
        for _ in range(3):                         # three extra stages
            pyramid.append(nn.Conv1d(c, c * 2, 4, stride=2, padding=1,
                                     bias=False))  # L → L/2
            pyramid.append(nn.BatchNorm1d(c * 2))
            pyramid.append(nn.ReLU(True))
            c *= 2
        self.pyramid = nn.Sequential(*pyramid)     # final c = base_channels*8


         # ── Dilated residual stack at the coarsest scale ───────────────────
        dilations = [1, 2, 4, 8]
        self.res_stack = nn.Sequential(
            *[ResBlock1D(c, d) for d in dilations]
        )

        # ── Adaptive pooling to H·W tokens ─────────────────────────────────
        self.apool = nn.AdaptiveAvgPool1d(self.n_tokens)   # (B,c,n_tokens)

        # ── Linear projection (per token) ─────────────────────────────────-
        self.proj = nn.Linear(c, latent_channels)

    # --------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, L)
        returns z: (B, C, H, W)
        """
        y = self.stem(x)           # (B, 64, L/2)
        y = self.pyramid(y)        # (B, c, L/16)
        y = self.res_stack(y)      # (B, c, L/16)
        y = self.apool(y)          # (B, c, H·W)

        # (B, c, n_tokens) → (B, n_tokens, c)
        y = y.permute(0, 2, 1).contiguous()
        y = self.proj(y)           # (B, n_tokens, C)
        # → (B, C, n_tokens)
        y = y.permute(0, 2, 1)
        B, C, _ = y.shape
        H, W = self.grid_size
        z = y.view(B, C, H, W)
        return z


# ────────────────────────────────────────────────────────────────────────────
# quick sanity check
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, L = 4, 1024               # batch size, light‑curve length
    x = torch.randn(B, 1, L)
    encoder = LightcurveEncoder1D()
    z = encoder(x)
    print("latent shape:", z.shape)  # → (4, 256, 8, 8) - (B, C, H, W)
