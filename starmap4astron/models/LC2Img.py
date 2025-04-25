import torch
import torch.nn as nn
from .encoder import LightcurveEncoder1D
from .decoder import Decoder2D

class LC2Img(nn.Module):
    def __init__(self,
                 *,
                 # ── Latent/grid parameters ───────────────────────────
                 latent_channels: int = 256,
                 grid_size: tuple[int, int] = (8, 8),

                 # ── Encoder-specific hyperparams ─────────────────────
                 in_channels: int = 1,
                 base_channels: int = 64,
                 num_pyramid: int = 3,
                 use_residuals: bool = True,
                 res_dilations: list[int] = [1, 2, 4, 8],

                 # ── Decoder/image parameters ─────────────────────────
                 img_size: int = 256,
                 decoder_base_channels: int | None = None
                 ):
        """
        LC2Img backbone:

        Args:
            latent_channels: number of channels in the latent grid
            grid_size: (H, W) size of latent grid
            in_channels: input LC channels (usually 1)
            base_channels: width of stem conv
            num_pyramid: how many /2 conv stages before pooling
            use_residuals: whether to include dilated ResBlocks
            res_dilations: list of dilations for each ResBlock
            img_size: output image height/width
            decoder_base_channels: channels for first upsample;
                                   defaults to latent_channels//2
        """
        super().__init__()
        self.img_size = img_size

        # ── Encoder ─────────────────────────────────────────────
        self.encoder = LightcurveEncoder1D(
            latent_channels=latent_channels,
            grid_size=grid_size,
            in_channels=in_channels,
            base_channels=base_channels,
            num_pyramid=num_pyramid,
            use_residuals=use_residuals,
            res_dilations=res_dilations,
        )

        # ── Decoder ─────────────────────────────────────────────
        # Determine decoder base channels if not supplied
        dec_ch = decoder_base_channels or (latent_channels // 2)
        ups = int(torch.log2(torch.tensor(img_size // grid_size[0])).item())
        self.decoder = Decoder2D(
            in_channels=latent_channels,
            base_channels=dec_ch,
            out_channels=1,      # grayscale output
            num_upsamples=ups
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_channels, L)
        returns: (B, out_channels, img_size, img_size)
        """
        z = self.encoder(x)       # → (B, latent_channels, H, W)
        img = self.decoder(z)     # → (B, 1, img_size, img_size)
        return img


# Old version
# import torch
# import torch.nn as nn
# from .encoder_v2 import LightcurveEncoder1D
# from .decoder import Decoder2D

# class LC2Img(nn.Module):
#     def __init__(self,
#                  latent_channels: int = 256,
#                  grid_size: tuple[int,int] = (8,8),
#                  img_size: int = 256,
#                  base_channels: int = 64):
#         super().__init__()
#         self.img_size = img_size
#         # your existing encoder
#         self.encoder = LightcurveEncoder1D(
#             latent_channels=latent_channels,
#             grid_size=grid_size,
#             in_channels=1,
#             base_channels=base_channels
#         )
#         # build decoder so final size = img_size
#         # num_upsamples = log2(img_size / grid_size[0])
#         ups = int(torch.log2(torch.tensor(img_size // grid_size[0])).item())
#         self.decoder = Decoder2D(
#             in_channels=latent_channels,
#             base_channels=latent_channels//2,
#             out_channels=1,
#             num_upsamples=ups
#         )




#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: (B,1,L)
#         returns: (B,3,img_size,img_size)
#         """
#         z = self.encoder(x)       # → (B, C, H, W)
#         img = self.decoder(z)     # → (B, 3, img_size, img_size)
#         return img
