# import argparse
# import os
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.loggers import TensorBoardLogger
# import matplotlib.pyplot as plt
# import torch
# from trainers.lc2img_module import LC2ImgModule
# from torch.utils.data import DataLoader, random_split
# from data.dataset import StarryNPZDataset
# from pytorch_lightning.callbacks import Callback

# import json


# class LossHistory(Callback):
#     """Collect train & val losses so we can plot them after training."""
#     def __init__(self):
#         super().__init__()
#         self.train_losses, self.val_losses = [], []

#     def on_train_epoch_end(self, trainer, pl_module):
#         self.train_losses.append(
#             trainer.callback_metrics["train/loss"].detach().cpu().item()
#         )

#     def on_validation_epoch_end(self, trainer, pl_module):
#         # skip if val loop didn’t run (e.g., first epoch with no val set yet)
#         if "val/loss" in trainer.callback_metrics:
#             self.val_losses.append(
#                 trainer.callback_metrics["val/loss"].detach().cpu().item()
#             )


# def main():
#     parser = argparse.ArgumentParser(description="Train LC2Img model")
#     parser.add_argument('--data_dir', type=str, required=True,
#                         help='Path to directory with light-curve + image pairs')
#     parser.add_argument('--val_data_dir', type=str, default=None,
#                         help='Optional: Path to validation data directory')
#     parser.add_argument('--val_split', type=float, default=0.1,
#                         help='Validation split ratio (used if val_data_dir not provided)')
#     parser.add_argument('--batch_size', type=int, default=32)
#     parser.add_argument('--img_size', type=int, default=64)
#     parser.add_argument('--latent_channels', type=int, default=256)
#     parser.add_argument('--grid_h', type=int, default=8)
#     parser.add_argument('--grid_w', type=int, default=8)
#     parser.add_argument('--base_channels', type=int, default=64)
#     parser.add_argument('--lr', type=float, default=1e-4)
#     parser.add_argument('--max_epochs', type=int, default=100)
#     parser.add_argument('--gpus', type=int, default=1)
#     parser.add_argument('--output_dir', type=str, default='output',
#                         help='Directory to save models and plots')
#     parser.add_argument('--mask_corners', action='store_true',help='If set, ignore corner pixels (only compute loss inside inscribed circle)')
#     args = parser.parse_args()

#     # Create output directory if it doesn't exist
#     os.makedirs(args.output_dir, exist_ok=True)

#     # Instantiate LightningModule with args
#     model = LC2ImgModule(
#         lr=args.lr,
#         latent_channels=args.latent_channels,
#         grid_size=(args.grid_h, args.grid_w),
#         img_size=args.img_size,
#         base_channels=args.base_channels,
#         mask_corners=args.mask_corners
#     )

#     # Build DataLoader for training dataset
#     train_dataset = StarryNPZDataset(
#         path=args.data_dir,
#         lc_key='flux',
#         img_key='image',
#         img_size=args.img_size,
#     )

#     # Setup validation data
#     if args.val_data_dir:
#         # Use separate validation dataset
#         val_dataset = StarryNPZDataset(
#             path=args.val_data_dir,
#             lc_key='flux',
#             img_key='image',
#             img_size=args.img_size,
#         )
#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=args.batch_size,
#             shuffle=True,
#             num_workers=4,
#             persistent_workers=True,
#             pin_memory=True if torch.cuda.is_available() else False
#         )
#         val_loader = DataLoader(
#             val_dataset,
#             batch_size=args.batch_size,
#             shuffle=False,
#             num_workers=4,
#             persistent_workers=True,
#             pin_memory=True if torch.cuda.is_available() else False
#         )
#     else:
#         # Split training data for validation
#         val_size = int(len(train_dataset) * args.val_split)
#         train_size = len(train_dataset) - val_size
#         train_subset, val_subset = random_split(
#             train_dataset, [train_size, val_size]
#         )

#         train_loader = DataLoader(
#             train_subset,
#             batch_size=args.batch_size,
#             shuffle=True,
#             num_workers=4,
#             persistent_workers=True,
#             pin_memory=True if torch.cuda.is_available() else False
#         )
#         val_loader = DataLoader(
#             val_subset,
#             batch_size=args.batch_size,
#             shuffle=False,
#             num_workers=4,
#             persistent_workers=True,
#             pin_memory=True if torch.cuda.is_available() else False
#         )

#     # 3️⃣  Callbacks & logger
#     loss_history = LossHistory()
#     ckpt_cb = ModelCheckpoint(
#         dirpath=os.path.join(args.output_dir, "checkpoints"),
#         filename="{epoch:03d}-{val_loss:.4f}",
#         save_top_k=3,
#         monitor="val/loss",
#         mode="min",
#     )
#     logger = TensorBoardLogger(save_dir=args.output_dir, name="tb_logs")


#     # Initialize trainer
#     trainer = pl.Trainer(
#         max_epochs=args.max_epochs,
#         accelerator="gpu" if torch.cuda.is_available() else "cpu",
#         devices=args.gpus if torch.cuda.is_available() else 1,
#         callbacks=[loss_history, ckpt_cb],
#         logger=logger,
#         log_every_n_steps=5
#     )

# #         #For CPU
# #     trainer = pl.Trainer(
# #     max_epochs=args.max_epochs,
# #     accelerator="cpu",
# #     devices=1,
# #     log_every_n_steps=5,
# #     callbacks=[loss_history, ckpt_cb],
# #     logger=logger,
# # )

#     # Train the model
#     trainer.fit(model, train_loader, val_loader)

#     # Save the final model
#     final_model_path = os.path.join(args.output_dir, 'final_model.pt')
#     torch.save(model.state_dict(), final_model_path)
#     print(f"Final model saved to {final_model_path}")

#     if loss_history.train_losses and loss_history.val_losses:
#         n=len(loss_history.train_losses)
#         epochs = range(1, n + 1)
#         plt.figure(figsize=(8,5))
#         plt.plot(epochs, loss_history.train_losses, label="train")
#         plt.plot(epochs, loss_history.val_losses[:n],   label="val")
#         plt.xlabel("Epoch")
#         plt.ylabel("Loss")
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(os.path.join(args.output_dir, "loss_curve.png"))
#         plt.close()
#         print("Epochs:", epochs)
#         print("Train losses:", loss_history.train_losses)
#         print("Val losses:", loss_history.val_losses)
#         print("Saved loss curve ➜", os.path.join(args.output_dir, "loss_curve.png"))

#         # Save loss values to JSON file for later use
#         loss_data = {
#             "epochs": list(epochs),
#             "train_losses": loss_history.train_losses,
#             "val_losses": loss_history.val_losses
#         }
#         with open(os.path.join(args.output_dir, "loss_data.json"), "w") as f:
#             json.dump(loss_data, f, indent=4)
#         print("Loss data saved to JSON file ➜", os.path.join(args.output_dir, "loss_data.json"))


#     print("Training completed!")

# def load_model_for_inference(model_path, latent_channels=256, grid_size=(8, 8),
#                             img_size=64, base_channels=64):
#     """
#     Load a trained model for inference

#     Args:
#         model_path: Path to saved model state dict

#     Returns:
#         Loaded model ready for inference
#     """
#     model = LC2ImgModule(
#         latent_channels=latent_channels,
#         grid_size=grid_size,
#         img_size=img_size,
#         base_channels=base_channels
#     )
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#     return model

# if __name__ == '__main__':
#     main()
