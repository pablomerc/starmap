import torch
import torch.nn as nn
from .encoder import LightcurveEncoder1D
from .decoder import Decoder2D

class LC2Img(nn.Module):
    def __init__(self,
                 latent_channels: int = 256,
                 grid_size: tuple[int,int] = (8,8),
                 img_size: int = 256,
                 base_channels: int = 64):
        super().__init__()
        self.img_size = img_size
        # your existing encoder
        self.encoder = LightcurveEncoder1D(
            latent_channels=latent_channels,
            grid_size=grid_size,
            in_channels=1,
            base_channels=base_channels
        )
        # build decoder so final size = img_size
        # num_upsamples = log2(img_size / grid_size[0])
        ups = int(torch.log2(torch.tensor(img_size // grid_size[0])).item())
        self.decoder = Decoder2D(
            in_channels=latent_channels,
            base_channels=latent_channels//2,
            out_channels=1,
            num_upsamples=ups
        )




    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,1,L)
        returns: (B,3,img_size,img_size)
        """
        z = self.encoder(x)       # → (B, C, H, W)
        img = self.decoder(z)     # → (B, 3, img_size, img_size)
        return img
