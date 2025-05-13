#!/usr/bin/env bash
# tuning-bash.sh
# -------------------------------------------------------------------
# Simple grid-search hyperparameter tuning for LC2Img via train.py
# -------------------------------------------------------------------

set -euo pipefail

# ────────────────────────────────────────────────────────────────────
# 1) Activate your conda env (adjust if needed)
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH" ]; then
  source "$CONDA_SH"
  conda activate starmap4astro-env
else
  echo "ERROR: conda.sh not found at $CONDA_SH" >&2
  exit 1
fi

# ────────────────────────────────────────────────────────────────────
# 2) Project & data paths
PROJECT_DIR="/Users/pablom.perez/Desktop/MIT-PhD-macbook/starmap/starmap4astron"
TRAIN_SCRIPT="$PROJECT_DIR/train.py"
DATA_DIR="$PROJECT_DIR/synthetic_data_output/dataset2/synthetic_starry_data_part_0.npz"

# ────────────────────────────────────────────────────────────────────
# 3) Fixed parameters
VAL_SPLIT=0.1
BATCH_SIZE=32
IMG_SIZE=256
GRID_H=8
GRID_W=8
NUM_PYRAMID=3
RES_DILATIONS="1 2 4 8"
GPUS=1
MAX_EPOCHS=50

# ────────────────────────────────────────────────────────────────────
# 4) Hyperparameter grid
LRS=(1e-4 5e-4 1e-3)
# LRS=(1e-4 5e-4)
LATENTS=(128 256 512)
# LATENTS=(128)
BASES=(32 64 128)
# BASES=(32)
USE_RESIDUALS=(true false)
# USE_RESIDUALS=(false)
# ────────────────────────────────────────────────────────────────────
# 5) Loop over combinations
for LR in "${LRS[@]}"; do
  for LAT in "${LATENTS[@]}"; do
    for BASE in "${BASES[@]}"; do
      for USE_RES in "${USE_RESIDUALS[@]}"; do

        TAG="lr${LR}_lat${LAT}_base${BASE}_res${USE_RES}"
        OUTDIR="$PROJECT_DIR/hpam_tuning_output/$TAG"
        mkdir -p "$OUTDIR"

        # Build the command
        CMD=(python "$TRAIN_SCRIPT"
          --data_dir "$DATA_DIR"
          --val_split $VAL_SPLIT
          --batch_size $BATCH_SIZE
          --img_size $IMG_SIZE
          --latent_channels $LAT
          --grid_h $GRID_H
          --grid_w $GRID_W
          --base_channels $BASE
          --num_pyramid $NUM_PYRAMID
          --res_dilations $RES_DILATIONS
        )
        # optionally include the residuals flag
        if [ "$USE_RES" = "true" ]; then
          CMD+=(--use_residuals)
        fi
        # continue fixed args
        CMD+=(
          --lr $LR
          --max_epochs $MAX_EPOCHS
          --gpus $GPUS
          --output_dir "$OUTDIR"
        )

        echo
        echo "────────────────────────────────────────────────────────"
        echo "Running trial: $TAG"
        echo "Command: ${CMD[*]}"
        echo "Output ➜ $OUTDIR"
        echo "────────────────────────────────────────────────────────"
        # Run it
        "${CMD[@]}"

      done
    done
  done
done

echo
echo "All trials finished! Check each $PROJECT_DIR/hpam_tuning_output/<tag>/ for results."
