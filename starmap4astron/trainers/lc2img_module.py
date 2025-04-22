import pytorch_lightning as pl
import torch.nn.functional as F
# from models.lc2img import LC2Img
from models import LC2Img
import torch

class LC2ImgModule(pl.LightningModule):
    """
    LightningModule wrapping LC2Img with training logic.
    """
    def __init__(self,
                 lr: float = 1e-4,
                 latent_channels: int = 256,
                 grid_size: tuple[int,int] = (8, 8),
                 img_size: int = 64,
                 base_channels: int = 64):
        super().__init__()
        self.save_hyperparameters()
        self.model = LC2Img(
            latent_channels=self.hparams.latent_channels,
            grid_size=self.hparams.grid_size,
            img_size=self.hparams.img_size,
            base_channels=self.hparams.base_channels
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        lc, img = batch
        pred = self(lc)
        loss = F.l1_loss(pred, img)
        self.log('train/loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
