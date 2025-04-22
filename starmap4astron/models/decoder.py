import torch
import torch.nn as nn

class Decoder2D(nn.Module):
    """
    Upsamples a small latent grid (C×H×W) to a full image (3×H_out×W_out).
    """
    def __init__(self,
                 in_channels: int = 256,
                 base_channels: int = 128,
                 out_channels: int = 1,
                 num_upsamples: int = 3):
        super().__init__()
        layers = []
        c = in_channels
        # each loop doubles spatial size
        for i in range(num_upsamples):
            layers += [
                nn.ConvTranspose2d(c, base_channels // (2**i),
                                   kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_channels // (2**i)),
                nn.ReLU(True),
            ]
            c = base_channels // (2**i)

        # final conv to RGB
        layers += [
            nn.Conv2d(c, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()   # assume your images are scaled to [-1,1]
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, C, H, W)  e.g. (B,256,8,8)
        returns img: (B, 3, H_out, W_out)  e.g. (B,3,64,64)
        """
        return self.net(z)
