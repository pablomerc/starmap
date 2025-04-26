
import argparse
import os
import json

import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split

from trainers.lc2img_module import LC2ImgModule
from data.dataset import StarryNPZDataset


class LossHistory(Callback):
    """Collect train & val losses so we can plot them after training."""
    def __init__(self):
        super().__init__()
        self.train_losses, self.val_losses = [], []

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_losses.append(
            trainer.callback_metrics["train/loss"].detach().cpu().item()
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        if "val/loss" in trainer.callback_metrics:
            self.val_losses.append(
                trainer.callback_metrics["val/loss"].detach().cpu().item()
            )


def main():
    parser = argparse.ArgumentParser(description="Train LC2Img model")
    # Data & splitting
    parser.add_argument('--data_dir',      type=str,   required=True,
                        help='Path to directory with light-curve + image pairs')
    parser.add_argument('--val_data_dir',  type=str,   default=None,
                        help='Optional: Path to validation data directory')
    parser.add_argument('--val_split',     type=float, default=0.1,
                        help='Validation split ratio (if no separate val dir)')
    # DataLoader
    parser.add_argument('--batch_size',    type=int,   default=32)
    # Model / encoder hyperparameters
    parser.add_argument('--img_size',          type=int,   default=64)
    parser.add_argument('--latent_channels',   type=int,   default=256)
    parser.add_argument('--grid_h',            type=int,   default=8)
    parser.add_argument('--grid_w',            type=int,   default=8)
    parser.add_argument('--base_channels',     type=int,   default=64,
                        help='Width of the stem conv')
    parser.add_argument('--num_pyramid',       type=int,   default=3,
                        help='Number of extra down-sampling conv stages')
    parser.add_argument('--use_residuals',     action='store_true',
                        help='Include dilated ResBlock1D stack in encoder')
    parser.add_argument('--res_dilations',     type=int,   nargs='+',
                        default=[1, 2, 4, 8],
                        help='Dilation rates for each ResBlock1D')
    parser.add_argument('--mask_corners',      action='store_true',
                        help='Ignore corner pixels (compute loss inside inscribed circle)')
    # Training
    parser.add_argument('--lr',            type=float, default=1e-4)
    parser.add_argument('--max_epochs',    type=int,   default=100)
    parser.add_argument('--gpus',          type=int,   default=1)
    parser.add_argument('--output_dir',    type=str,   default='output',
                        help='Directory to save models and plots')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Save full config for later inference ────────────────────────────
    import json
    cfg_path = os.path.join(args.output_dir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    print(f"Saved training config ➜ {cfg_path}")

    # Instantiate LightningModule with encoder params
    model = LC2ImgModule(
        lr               = args.lr,
        latent_channels  = args.latent_channels,
        grid_size        = (args.grid_h, args.grid_w),
        img_size         = args.img_size,
        base_channels    = args.base_channels,
        num_pyramid      = args.num_pyramid,
        use_residuals    = args.use_residuals,
        res_dilations    = args.res_dilations,
        mask_corners     = args.mask_corners,
    )

    # Prepare datasets & loaders
    full_dataset = StarryNPZDataset(
        path      = args.data_dir,
        lc_key    = 'flux',
        img_key   = 'image',
        img_size  = args.img_size,
    )

    if args.val_data_dir:
        val_dataset = StarryNPZDataset(
            path     = args.val_data_dir,
            lc_key   = 'flux',
            img_key  = 'image',
            img_size = args.img_size,
        )
        train_dataset, val_dataset = full_dataset, val_dataset
    else:
        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available()
    )

    # Callbacks & logger
    loss_history = LossHistory()
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="{epoch:03d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val/loss",
        mode="min",
    )
    logger = TensorBoardLogger(save_dir=args.output_dir, name="tb_logs")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.gpus if torch.cuda.is_available() else 1,
        callbacks=[loss_history, ckpt_cb],
        logger=logger,
        log_every_n_steps=5,
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Save final weights
    final_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")

    # Plot loss curve
    if loss_history.train_losses and loss_history.val_losses:
        n=len(loss_history.train_losses)
        epochs = list(range(1, len(loss_history.train_losses) + 1))
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, loss_history.train_losses, label="train")
        plt.plot(epochs, loss_history.val_losses[:n],   label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        curve_path = os.path.join(args.output_dir, "loss_curve.png")
        plt.savefig(curve_path)
        plt.close()
        print(f"Saved loss curve ➜ {curve_path}")

        # Save raw loss data
        loss_data = {
            "epochs":       epochs,
            "train_losses": loss_history.train_losses,
            "val_losses":   loss_history.val_losses,
        }
        json_path = os.path.join(args.output_dir, "loss_data.json")
        with open(json_path, "w") as f:
            json.dump(loss_data, f, indent=4)
        print(f"Loss data saved ➜ {json_path}")
    print("Training completed!")

    # Save one example + reconstruction
    model.eval()
    # grab one batch
    xb, yb = next(iter(val_loader))  # xb: light curve, yb: image
    xb = xb.to(model.device)
    with torch.no_grad():
        ypred = model(xb)  # shape: (B, C, H, W)

    # only keep the first example
    gt_img = yb[0].cpu().squeeze().numpy()
    recon_img = ypred[0].cpu().squeeze().numpy()

    # ensure output_dir exists
    example_dir = os.path.join(args.output_dir, "example")
    os.makedirs(example_dir, exist_ok=True)

    # save with matplotlib
    for arr, name in [(gt_img, "sample_input.png"),
                    (recon_img, "sample_recon.png")]:
        plt.figure(figsize=(4,4))
        plt.imshow(arr, cmap='gray', origin='lower')
        plt.axis('off')
        plt.tight_layout(pad=0)
        path = os.path.join(example_dir, name)
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved {name} ➜ {path}")





def load_model_for_inference(model_path,
                             latent_channels=256,
                             grid_size=(8, 8),
                             img_size=64,
                             base_channels=64,
                             num_pyramid=3,
                             use_residuals=True,
                             res_dilations=[1,2,4,8],
                             mask_corners=False):
    """
    Load a trained model for inference.

    Returns the LC2ImgModule ready to run.
    """
    model = LC2ImgModule(
        latent_channels  = latent_channels,
        grid_size        = grid_size,
        img_size         = img_size,
        base_channels    = base_channels,
        num_pyramid      = num_pyramid,
        use_residuals    = use_residuals,
        res_dilations    = res_dilations,
        mask_corners     = mask_corners,
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


if __name__ == '__main__':
    main()
