import argparse
import pytorch_lightning as pl
from trainers.lc2img_module import LC2ImgModule
from torch.utils.data import DataLoader
from data.dataset import StarryNPZDataset

def main():
    parser = argparse.ArgumentParser(description="Train LC2Img model")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to directory with light-curve + image pairs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--latent_channels', type=int, default=256)
    parser.add_argument('--grid_h', type=int, default=8)
    parser.add_argument('--grid_w', type=int, default=8)
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--gpus', type=int, default=1)
    args = parser.parse_args()

    # Instantiate LightningModule with args
    model = LC2ImgModule(
        lr=args.lr,
        latent_channels=args.latent_channels,
        grid_size=(args.grid_h, args.grid_w),
        img_size=args.img_size,
        base_channels=args.base_channels
    )

    # Build DataLoader
    dataset = StarryNPZDataset(
    path=args.data_dir,   # either one .npz or a folder of them
    lc_key='flux',
    img_key='image',
    img_size=args.img_size,
)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=4)

    # Configure and run Trainer
    # trainer = pl.Trainer(
    #     max_epochs=args.max_epochs,
    #     devices=args.gpus,
    #     # Lightning will automatically call training_step, forward, etc.
    # )

    #For CPU
    trainer = pl.Trainer(
    max_epochs=args.max_epochs,
    accelerator="cpu",
    devices=1,
)


    trainer.fit(model, loader)


if __name__ == '__main__':
    main()
