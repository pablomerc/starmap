import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchmetrics.functional import structural_similarity_index_measure as ssim
from models import LC2Img


class LC2ImgModule(pl.LightningModule):
    """
    LightningModule wrapping LC2Img with optional Perceptual / SSIM loss.
    """
    def __init__(self,
                 lr: float = 1e-4,
                 latent_channels: int = 256,
                 grid_size: tuple[int, int] = (8, 8),
                 img_size: int = 64,
                 base_channels: int = 64,
                 num_pyramid: int = 3,
                 use_residuals: bool = True,
                 res_dilations: list[int] = [1, 2, 4, 8],
                 mask_corners: bool = True,
                 # ───────────────────────────────────────────────
                 use_perceptual_loss: bool = False,
                 use_ssim_loss: bool = False,
                 lambda_perc: float = 0.1,
                 lambda_ssim: float = 0.1):
        super().__init__()
        self.save_hyperparameters()

        # ── core model ──────────────────────────────────────
        self.model = LC2Img(
            latent_channels=self.hparams.latent_channels,
            grid_size=self.hparams.grid_size,
            img_size=self.hparams.img_size,
            base_channels=self.hparams.base_channels,
            num_pyramid=self.hparams.num_pyramid,
            use_residuals=self.hparams.use_residuals,
            res_dilations=self.hparams.res_dilations
        )

        # ── perceptual feature extractor (frozen) ───────────
        if self.hparams.use_perceptual_loss:
            print("Using VGG16 for perceptual loss")
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            # relu2_2 features (up to index 16)
            self.perc_net = nn.Sequential(*list(vgg.features)[:16]).eval()
            for p in self.perc_net.parameters():
                p.requires_grad_(False)

        if self.hparams.use_ssim_loss:
            print("Using SSIM loss")


        # ── circular mask buffer (unchanged) ────────────────
        if self.hparams.mask_corners:
            H = W = self.hparams.img_size
            yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            cx = cy = W // 2
            circle = ((xx - cx) ** 2 + (yy - cy) ** 2 <= min(cx, cy) ** 2).float()
            self.register_buffer('circle_mask', circle[None, None])
        else:
            self.register_buffer('circle_mask', None)

    # ── helpers ─────────────────────────────────────────────
    def _apply_mask(self, x):
        if self.circle_mask is None:
            return x
        return x * self.circle_mask

    def _pixel_loss(self, pred, target):
        return F.l1_loss(self._apply_mask(pred), self._apply_mask(target))

    def _perceptual_loss(self, pred, target):
        """L1 distance in VGG feature space (expects 3-channel inputs)."""
        # duplicate grayscale → RGB and normalise to Imagenet statistics
        def _preproc(t):
            t = t.repeat(1, 3, 1, 1)           # (B, 3, H, W)
            mean = torch.tensor([0.485, 0.456, 0.406], device=t.device)[None, :, None, None]
            std  = torch.tensor([0.229, 0.224, 0.225], device=t.device)[None, :, None, None]
            return (t + 1) / 2.0   # [-1,1] → [0,1] first
        f_pred = self.perc_net(_preproc(pred))
        f_tgt  = self.perc_net(_preproc(target))
        return F.l1_loss(f_pred, f_tgt)

    # ── Lightning hooks ────────────────────────────────────
    def forward(self, x):
        return self.model(x)

    def _total_loss(self, pred, target):
        loss = self._pixel_loss(pred, target)

        if self.hparams.use_perceptual_loss:
            loss += self.hparams.lambda_perc * self._perceptual_loss(pred, target)

        if self.hparams.use_ssim_loss:
            # SSIM returns [−1,1];  we convert to a *dissimilarity* term
            ssim_score = ssim(pred, target, data_range=2.0)  # inputs in [-1,1] ⇒ range=2
            loss += self.hparams.lambda_ssim * (1 - ssim_score)

        return loss

    def training_step(self, batch, _):
        lc, img = batch
        pred = self(lc)
        loss = self._total_loss(pred, img)
        self.log_dict({'train/loss': loss}, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        lc, img = batch
        pred = self(lc)
        loss = self._total_loss(pred, img)
        self.log_dict({'val/loss': loss}, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
