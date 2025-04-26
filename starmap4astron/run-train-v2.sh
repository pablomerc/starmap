#!/bin/bash
# filepath: train_model.sh

# ────────────────────────────────────────────────────────────────────────────
# Activate conda environment first
CONDA_SH="/pdo/users/pablomer/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH" ]; then
  source "$CONDA_SH"
  conda activate starmap4astro-env
else
  echo "ERROR: conda.sh not found at $CONDA_SH"
  exit 1
fi

# ────────────────────────────────────────────────────────────────────────────
# Required parameters
DATA_DIR="/pdo/users/pablomer/starmap/starmap4astron/synthetic_data_output/dataset2/synthetic_starry_data_part_0.npz"

# ────────────────────────────────────────────────────────────────────────────
# Optional parameters with their default values
VAL_DATA_DIR=""               # Leave empty for automatic train/val split
VAL_SPLIT=0.1                 # Validation split ratio (if no val dir)
BATCH_SIZE=32                 # Batch size for training
IMG_SIZE=64                   # Output image size
LATENT_CHANNELS=256           # Number of latent channels
GRID_H=16                      # Grid height
GRID_W=16                      # Grid width
BASE_CHANNELS=32              # Number of base channels
LR=1e-3                       # Learning rate
MAX_EPOCHS=500                # Maximum number of epochs
GPUS=1                        # Number of GPUs to use
OUTPUT_DIR="./output"         # Directory to save models and plots

# ── New encoder hyper-parameters ───────────────────────────────────────────
NUM_PYRAMID=3                 # how many extra down-sampling conv stages
USE_RESIDUALS=true           # whether to include the dilated ResBlocks
RES_DILATIONS="1 2 4 8"       # list of dilations for each ResBlock

# ── Masking option ─────────────────────────────────────────────────────────
MASK_CORNERS=false            # set to true to pass --mask_corners

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build command with all parameters
CMD="python /pdo/users/pablomer/starmap/starmap4astron/train.py \
  --data_dir \"$DATA_DIR\" \
  --batch_size $BATCH_SIZE \
  --img_size $IMG_SIZE \
  --latent_channels $LATENT_CHANNELS \
  --grid_h $GRID_H \
  --grid_w $GRID_W \
  --base_channels $BASE_CHANNELS \
  --lr $LR \
  --max_epochs $MAX_EPOCHS \
  --gpus $GPUS \
  --output_dir \"$OUTPUT_DIR\" \
  --val_split $VAL_SPLIT \
  --num_pyramid $NUM_PYRAMID"

# Validation directory if set
if [ -n "$VAL_DATA_DIR" ]; then
  CMD="$CMD --val_data_dir \"$VAL_DATA_DIR\""
fi

# Residual stack flag
if [ "$USE_RESIDUALS" = true ]; then
  CMD="$CMD --use_residuals"
fi

# Dilation rates
CMD="$CMD --res_dilations $RES_DILATIONS"

# Mask-corners flag
if [ "$MASK_CORNERS" = true ]; then
  CMD="$CMD --mask_corners"
fi

# Print & run
echo "Running: $CMD"
eval $CMD
