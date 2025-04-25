import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv1d(channels, channels, 3,
                               padding=pad, dilation=dilation, bias=False)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3,
                               padding=pad, dilation=dilation, bias=False)
        self.bn2   = nn.BatchNorm1d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class LightcurveEncoder1D(nn.Module):
    """
    1-D light-curve → (C, H, W) latent grid.

    Args
    ----
    latent_channels : int
        Number of output channels C.
    grid_size : (H, W)
        Spatial size of the output grid.
    in_channels : int
        Input channels (1 for flux).
    base_channels : int
        Width of the first convolution.
    num_pyramid : int
        Number of extra down-sampling conv stages (each halves length, doubles channels).
    use_residuals : bool
        Whether to include the dilated ResBlock1D stack.
    res_dilations : list[int]
        Dilation rates for each ResBlock1D (if use_residuals=True).
    """
    def __init__(self,
                 latent_channels: int = 256,
                 grid_size: tuple[int, int] = (8, 8),
                 in_channels: int = 1,
                 base_channels: int = 64,
                 num_pyramid: int = 3,
                 use_residuals: bool = True,
                 res_dilations: list[int] = [1, 2, 4, 8]):
        super().__init__()

        H, W = grid_size
        self.grid_size = (H, W)
        self.latent_channels = latent_channels
        self.n_tokens = H * W

        # ── Stem ───────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, 7,
                      stride=2, padding=3, bias=False),  # L → L/2
            nn.BatchNorm1d(base_channels),
            nn.ReLU(True),
        )

        # ── Pyramid down-sampling ──────────────────────────────────────────
        pyramid = []
        c = base_channels
        for _ in range(num_pyramid):
            pyramid += [
                nn.Conv1d(c, c * 2, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(c * 2),
                nn.ReLU(True),
            ]
            c *= 2
        self.pyramid = nn.Sequential(*pyramid)
        self.out_channels = c  # channels at coarsest scale

        # ── Optional residual stack ────────────────────────────────────────
        self.use_residuals = use_residuals
        if use_residuals:
            self.res_stack = nn.Sequential(
                *[ResBlock1D(c, d) for d in res_dilations]
            )

        # ── Token pooling + projection ─────────────────────────────────────
        self.apool = nn.AdaptiveAvgPool1d(self.n_tokens)  # (B, c, H·W)
        self.proj  = nn.Linear(self.out_channels, latent_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_channels, L)
        returns z: (B, C, H, W)
        """
        y = self.stem(x)           # → (B, base, L/2)
        y = self.pyramid(y)        # → (B, c, L/(2·2^num_pyramid))
        if self.use_residuals:
            y = self.res_stack(y)  # → (B, c, same)
        y = self.apool(y)          # → (B, c, H·W)

        # (B, c, tokens) → (B, tokens, c)
        y = y.permute(0, 2, 1).contiguous()
        y = self.proj(y)           # → (B, tokens, C)

        # → (B, C, tokens) → (B, C, H, W)
        y = y.permute(0, 2, 1)
        B, C, _ = y.shape
        H, W = self.grid_size
        return y.view(B, C, H, W)


if __name__ == "__main__":
    # test both variants
    for use_res in [False, True]:
        enc = LightcurveEncoder1D(
            latent_channels=256,
            grid_size=(8, 8),
            in_channels=1,
            base_channels=32,
            num_pyramid=2,
            use_residuals=use_res,
            res_dilations=[1, 2]
        )
        x = torch.randn(4, 1, 1024)
        z = enc(x)
        print(f"use_residuals={use_res} → z.shape = {z.shape}")
        # should print torch.Size([4, 256, 8, 8]) for both
