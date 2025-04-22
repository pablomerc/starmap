#!/usr/bin/env bash
set -euo pipefail

# point to your conda.sh
CONDA_SH="/Users/pablom.perez/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH" ]; then
  source "$CONDA_SH"
  conda activate starmap4astro-env
else
  echo "ERROR: conda.sh not found at $CONDA_SH"
  exit 1
fi

# 2) Train the model
DATA_DIR=${1:-"/Users/pablom.perez/Desktop/MIT-PhD-macbook/starmap/starmap4astron/synthetic_data_output/dataset2/synthetic_starry_data_part_0.npz"}
BATCH_SIZE=32
IMG_SIZE=256
LATENT_CHANNELS=256
GRID_H=8
GRID_W=8
BASE_CHANNELS=64
LR=1e-4
MAX_EPOCHS=25
GPUS=1

echo "=== Training LC2Img on $DATA_DIR ==="
python train.py \
  --data_dir "$DATA_DIR" \
  --batch_size $BATCH_SIZE \
  --img_size $IMG_SIZE \
  --latent_channels $LATENT_CHANNELS \
  --grid_h $GRID_H \
  --grid_w $GRID_W \
  --base_channels $BASE_CHANNELS \
  --lr $LR \
  --max_epochs $MAX_EPOCHS \
  --gpus $GPUS

# 3) Locate the latest checkpoint
CKPT=$(ls lightning_logs/version_*/checkpoints/*.ckpt | tail -n1)
echo "=== Using checkpoint: $CKPT ==="

# 3) Locate the latest checkpoint
CKPT=$(ls lightning_logs/version_*/checkpoints/*.ckpt | tail -n1)
echo "=== Using checkpoint: $CKPT ==="

# 4) Pick the .npz (same logic as before)
if [ -d "$DATA_DIR" ]; then
  NPZ=$(ls "$DATA_DIR"/*.npz | head -n1)
elif [ -f "$DATA_DIR" ]; then
  NPZ="$DATA_DIR"
else
  echo "ERROR: DATA_DIR is not a valid file or directory"
  exit 1
fi
echo "=== Running inference on: $NPZ ==="

# 5) Call your standalone inference.py
python inference.py "$CKPT" "$NPZ" --out recon.png
echo "Saved recon.png"



echo "All done!"
