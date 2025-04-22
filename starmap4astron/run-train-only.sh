#!/bin/bash
# filepath: train_model.sh

# Activate conda environment first
CONDA_SH="/Users/pablom.perez/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH" ]; then
  source "$CONDA_SH"
  conda activate starmap4astro-env
else
  echo "ERROR: conda.sh not found at $CONDA_SH"
  exit 1
fi

# Required parameters
DATA_DIR="/Users/pablom.perez/Desktop/MIT-PhD-macbook/starmap/starmap4astron/synthetic_data_output/dataset2/synthetic_starry_data_part_0.npz" # CHANGE THIS to your actual data path

# Optional parameters with their default values
VAL_DATA_DIR=""              # Leave empty for automatic train/val split
VAL_SPLIT=0.1                # Validation split ratio (if VAL_DATA_DIR not provided)
BATCH_SIZE=32                # Batch size for training
IMG_SIZE=64                  # Output image size
LATENT_CHANNELS=256          # Number of latent channels
GRID_H=8                     # Grid height
GRID_W=8                     # Grid width
BASE_CHANNELS=64             # Number of base channels
LR=1e-4                      # Learning rate
MAX_EPOCHS=50               # Maximum number of epochs
GPUS=1                       # Number of GPUs to use
OUTPUT_DIR="./output"        # Directory to save models and plots

# Create output directory
mkdir -p $OUTPUT_DIR

# Build command with all parameters
CMD="python /Users/pablom.perez/Desktop/MIT-PhD-macbook/starmap/starmap4astron/train.py \
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
  --val_split $VAL_SPLIT"

# Add validation directory if provided
if [ ! -z "$VAL_DATA_DIR" ]; then
  CMD="$CMD --val_data_dir \"$VAL_DATA_DIR\""
fi

# Print the command
echo "Running: $CMD"

# Execute the command
eval $CMD
