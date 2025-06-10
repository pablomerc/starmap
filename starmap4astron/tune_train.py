# tune_train.py

import argparse
import os
import json

import optuna
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from trainers.lc2img_module import LC2ImgModule
from data.dataset import StarryNPZDataset


def objective(trial: optuna.Trial, args):
    # ── Suggest hyperparameters ────────────────────────────────────────────
    lr            = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    latent_ch     = trial.suggest_categorical("latent_channels", [128, 256, 512])
    base_ch       = trial.suggest_int("base_channels", 32, 128, step=32)
    num_pyramid   = trial.suggest_int("num_pyramid", 1, 4)
    use_res       = trial.suggest_categorical("use_residuals", [False, True])

    # ── Prepare a per-trial output directory ─────────────────────────────────
    trial_dir = os.path.join(args.output_dir, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)
    # (Optionally record the trial config)
    with open(os.path.join(trial_dir, "config.json"), "w") as fp:
        json.dump({
            "lr": lr,
            "latent_channels": latent_ch,
            "base_channels": base_ch,
            "num_pyramid": num_pyramid,
            "use_residuals": use_res,
            **vars(args)
        }, fp, indent=4)

    # ── Build the model ─────────────────────────────────────────────────────
    model = LC2ImgModule(
        lr               = lr,
        latent_channels  = latent_ch,
        grid_size        = (args.grid_h, args.grid_w),
        img_size         = args.img_size,
        base_channels    = base_ch,
        num_pyramid      = num_pyramid,
        use_residuals    = use_res,
        res_dilations    = args.res_dilations,
        mask_corners     = args.mask_corners,
    )

    # ── Prepare data loaders ─────────────────────────────────────────────────
    full_ds = StarryNPZDataset(
        path     = args.data_dir,
        lc_key   = "flux",
        img_key  = "image",
        img_size = args.img_size,
    )
    if args.val_data_dir:
        val_ds = StarryNPZDataset(
            path     = args.val_data_dir,
            lc_key   = "flux",
            img_key  = "image",
            img_size = args.img_size,
        )
        train_ds, val_ds = full_ds, val_ds
    else:
        val_size   = int(len(full_ds) * args.val_split)
        train_size = len(full_ds) - val_size
        train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        persistent_workers=False,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        persistent_workers=False,
        pin_memory=torch.cuda.is_available(),
    )

    # ── Callbacks & logger ──────────────────────────────────────────────────
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(trial_dir, "checkpoints"),
        filename="{epoch:03d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val/loss",
        mode="min",
    )
    logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name="optuna",
        version=f"trial_{trial.number}"
    )

    # ── Trainer ─────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs      = args.max_epochs,
        accelerator     = "gpu" if torch.cuda.is_available() else "cpu",
        devices         = args.gpus if torch.cuda.is_available() else 1,
        callbacks       = [ckpt_cb],
        logger          = logger,
        log_every_n_steps = 5,
    )

    # ── Run training ─────────────────────────────────────────────────────────
    trainer.fit(model, train_loader, val_loader)

    # ── Return the validation loss for Optuna to minimize ───────────────────
    return trainer.callback_metrics["val/loss"].item()


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for LC2Img with Optuna"
    )
    # data & splitting
    parser.add_argument("--data_dir",      type=str, required=True)
    parser.add_argument("--val_data_dir",  type=str, default=None)
    parser.add_argument("--val_split",     type=float, default=0.1)
    # loaders
    parser.add_argument("--batch_size",    type=int,   default=32)
    # model architecture defaults (will be overridden by Optuna in objective)
    parser.add_argument("--img_size",          type=int,   default=64)
    parser.add_argument("--grid_h",            type=int,   default=8)
    parser.add_argument("--grid_w",            type=int,   default=8)
    parser.add_argument("--res_dilations",     type=int, nargs="+", default=[1,2,4,8])
    parser.add_argument("--mask_corners",      action="store_true")
    # training
    parser.add_argument("--max_epochs",    type=int,   default=100)
    parser.add_argument("--gpus",          type=int,   default=1)
    # tuning
    parser.add_argument("--output_dir",    type=str,   default="optuna_output")
    parser.add_argument("--n_trials",      type=int,   default=20)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
    )
    study.optimize(lambda t: objective(t, args), n_trials=args.n_trials)

    print("Best hyperparameters found:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
