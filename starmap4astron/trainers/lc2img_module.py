import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from models import LC2Img
import torch.nn as nn

import torchvision.models as models
from torchmetrics.functional import structural_similarity_index_measure as ssim

class LC2ImgModule(pl.LightningModule):
    """
    LightningModule wrapping LC2Img with training logic.
    """
    def __init__(self,
                 lr: float = 1e-4,
                 latent_channels: int = 256,
                 grid_size: tuple[int,int] = (8, 8),
                 img_size: int = 64,
                 base_channels: int = 64,
                 num_pyramid: int = 3,
                 use_residuals: bool = True,
                 res_dilations: list[int] = [1,2,4,8],
                 mask_corners: bool = True,
                 use_perceptual_loss: bool = False,
                 use_ssim_loss: bool = False,
                 lambda_perc: float = 0.1,
                 lambda_ssim: float = 0.1):
        super().__init__()
        # register all args to self.hparams
        self.save_hyperparameters()

        # build the LC2Img model with encoder hyperparams
        self.model = LC2Img(
            latent_channels = self.hparams.latent_channels,
            grid_size       = self.hparams.grid_size,
            img_size        = self.hparams.img_size,
            base_channels   = self.hparams.base_channels,
            num_pyramid     = self.hparams.num_pyramid,
            use_residuals   = self.hparams.use_residuals,
            res_dilations   = self.hparams.res_dilations
        )

        # ── perceptual feature extractor (frozen) ───────────
        if self.hparams.use_perceptual_loss:
            print("[LC2ImgModule] Using perceptual loss")
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            # relu2_2 features (up to index 16)
            self.perc_net = nn.Sequential(*list(vgg.features)[:16]).eval()
            for p in self.perc_net.parameters():
                p.requires_grad_(False)




        # optionally build circular mask buffer
        if self.hparams.mask_corners:
            H = W = self.hparams.img_size
            yy, xx = torch.meshgrid(torch.arange(H),
                                     torch.arange(W),
                                     indexing='ij')
            cx, cy = W//2, H//2
            circle = ((xx-cx)**2 + (yy-cy)**2 <= min(cx,cy)**2).float()
            self.register_buffer('circle_mask',
                                 circle.unsqueeze(0).unsqueeze(0))
            print(f"[LC2ImgModule] circle_mask registered, "
                  f"shape: {self.circle_mask.shape}")
        else:
            self.register_buffer('circle_mask', None)
            print("[LC2ImgModule] No circle_mask registered; "
                  "using full-image loss")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _masked_loss(self, pred, target):
        if self.hparams.mask_corners and self.circle_mask is not None:
            m = self.circle_mask
            return F.l1_loss(pred*m, target*m, reduction='sum') / m.sum()
        else:
            return F.l1_loss(pred, target)

    def training_step(self, batch, batch_idx):
        lc, img = batch
        pred = self(lc)
        loss = self._masked_loss(pred, img)
        self.log('train/loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        lc, img = batch
        pred = self(lc)
        loss = self._masked_loss(pred, img)
        self.log('val/loss', loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        lc, img = batch
        pred = self(lc)
        loss = self._masked_loss(pred, img)
        self.log('test/loss', loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


# ### Old
# import pytorch_lightning as pl
# import torch.nn.functional as F
# # from models.lc2img import LC2Img
# from models import LC2Img
# import torch

# class LC2ImgModule(pl.LightningModule):
#     """
#     LightningModule wrapping LC2Img with training logic.
#     """
#     def __init__(self,
#                  lr: float = 1e-4,
#                  latent_channels: int = 256,
#                  grid_size: tuple[int,int] = (8, 8),
#                  img_size: int = 64,
#                  base_channels: int = 64,
#                  mask_corners: bool = True):
#         super().__init__()
#         self.save_hyperparameters()
#         self.model = LC2Img(
#             latent_channels=self.hparams.latent_channels,
#             grid_size=self.hparams.grid_size,
#             img_size=self.hparams.img_size,
#             base_channels=self.hparams.base_channels
#         )
#         # Only build the mask if requested
#         if self.hparams.mask_corners:

#             H = W = self.hparams.img_size
#             yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
#             cx, cy = W//2, H//2
#             circle = ((xx-cx)**2 + (yy-cy)**2 <= (min(cx,cy))**2).float()
#             # register so it moves to GPU/CPU with the model
#             self.register_buffer('circle_mask', circle.unsqueeze(0).unsqueeze(0))

#             print(f"[LC2ImgModule] circle_mask registered, shape: {self.circle_mask.shape}")

#         else:
#             self.register_buffer('circle_mask', None)
#             print("[LC2ImgModule] No circle_mask registered, will compute loss on all pixels")


#     def forward(self, x):
#         return self.model(x)

#     def _masked_loss(self, pred, target):
#         if self.hparams.mask_corners and self.circle_mask is not None:
#             m = self.circle_mask
#             return F.l1_loss(pred*m, target*m, reduction='sum') / m.sum()
#         else:
#             return F.l1_loss(pred, target)

#     def training_step(self, batch, batch_idx):
#         lc, img = batch
#         pred = self(lc)
#         loss = self._masked_loss(pred, img)
#         self.log('train/loss', loss, on_epoch=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         lc, img = batch
#         pred = self(lc)
#         loss = self._masked_loss(pred, img)
#         self.log('val/loss', loss,on_epoch=True)
#         return loss

#     def test_step(self, batch, batch_idx):
#         lc, img = batch
#         pred = self(lc)
#         loss = self._masked_loss(pred, img)
#         self.log('test/loss', loss,on_epoch=True)
#         return loss

#     def configure_optimizers(self):
#         return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
