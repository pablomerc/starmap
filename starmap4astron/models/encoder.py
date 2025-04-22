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

        # Stem: wide receptive field, stride 2 to shrink sequence length
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels,
                      kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(True)
        )

        # Four residual blocks with exponentially increasing dilation
        dilations = [1, 2, 4, 8]
        channels  = base_channels
        blocks = []
        for d in dilations:
            blocks.append(ResBlock1D(channels, dilation=d))
        self.backbone = nn.Sequential(*blocks)

        # Global aggregation
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # (B, C, 1) → (B, C)

        # FC to latent grid (flattened)
        grid_elems = latent_channels * grid_size[0] * grid_size[1]
        self.proj = nn.Linear(channels, grid_elems)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, L) light‑curve
        returns z: (B, C, H, W)
        """
        y = self.stem(x)              # (B, 64, L/2)
        y = self.backbone(y)          # (B, 64, L/2)
        y = self.global_pool(y).squeeze(-1)     # (B, 64)
        y = self.proj(y)                           # (B, C*H*W)
        C, H, W = self.latent_channels, *self.grid_size
        z = y.view(-1, C, H, W)
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
