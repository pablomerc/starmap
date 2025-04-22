#!/bin/bash

# Set parameters for training
DATA_DIR="/path/to/your/data"
VAL_DATA_DIR="/path/to/validation/data"
OUTPUT_DIR="./outputs/experiment2"
BATCH_SIZE=32
IMG_SIZE=64
MAX_EPOCHS=100
GPUS=1

# Run training script with validation data
python /Users/pablom.perez/Desktop/MIT-PhD-macbook/starmap/starmap4astron/train.py \
  --data_dir "$DATA_DIR" \
  --val_data_dir "$VAL_DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --img_size "$IMG_SIZE" \
  --max_epochs "$MAX_EPOCHS" \
  --gpus "$GPUS"
