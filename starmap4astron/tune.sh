#!/bin/bash
# filepath: tune.sh

# ────────────────────────────────────────────────────────────────────────────
# Base project directory
ROOT_DIR="/Users/pablom.perez/Desktop/MIT-PhD-macbook/starmap/starmap4astron"

# ────────────────────────────────────────────────────────────────────────────
# Activate conda environment (adjust if your conda is installed elsewhere)
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH" ]; then
  source "$CONDA_SH"
  conda activate starmap4astro-env
else
  echo "ERROR: conda.sh not found at $CONDA_SH"
  exit 1
fi

# ────────────────────────────────────────────────────────────────────────────
# User parameters (tweak as needed)
DATA_DIR="$ROOT_DIR/synthetic_data_output/dataset2/synthetic_starry_data_part_0.npz"
VAL_DATA_DIR=""               # leave empty to use --val_split
OUTPUT_DIR="$ROOT_DIR/optuna_output"
N_TRIALS=3                    # number of hyperparameter trials
MAX_EPOCHS=3                  # epochs per trial
GPUS=1                         # GPUs to use if available
BATCH_SIZE=32

# ────────────────────────────────────────────────────────────────────────────
# Build the tuning command, pointing at tune_train.py in project root
CMD=(python "$ROOT_DIR/tune_train.py"
  --data_dir "$DATA_DIR"
  --batch_size $BATCH_SIZE
  --max_epochs $MAX_EPOCHS
  --gpus $GPUS
  --output_dir "$OUTPUT_DIR"
  --n_trials $N_TRIALS
)

# validation split or explicit val dir
if [ -n "$VAL_DATA_DIR" ]; then
  CMD+=(--val_data_dir "$VAL_DATA_DIR")
else
  CMD+=(--val_split 0.1)
fi

# ────────────────────────────────────────────────────────────────────────────
# Run hyperparameter sweep
echo "Running hyperparameter sweep with Optuna:"
echo "${CMD[@]}"
"${CMD[@]}"

# ────────────────────────────────────────────────────────────────────────────
# Launch TensorBoard for visualization

# before launching TensorBoard in tune.sh
pkill -f "tensorboard.*--port 6006" || true


echo "Hyperparameter tuning complete."
echo "Launching TensorBoard on port 6006 to visualize results..."
tensorboard --logdir "$OUTPUT_DIR/optuna" --bind_all --port 6006 &

echo "TensorBoard is running at http://localhost:6006"
