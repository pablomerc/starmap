#!/usr/bin/env bash
# hpam_tune_starmap.sh
# ─────────────────────────────────────────────────────────────────────────────
# Grid-search hyper-parameter tuning for LC2Img (StarMap project)
# Uses the same flags as train_model.sh, plus the new loss options.
# ---------------------------------------------------------------------------

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# 0) Activate conda environment

CONDA_SH="/pdo/users/pablomer/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH" ]; then
  source "$CONDA_SH"
  conda activate sm4as-env
else
  echo "ERROR: conda.sh not found at $CONDA_SH"
  exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# 1) Paths
PROJECT_DIR="/pdo/users/pablomer/starmap/starmap4astron"
TRAIN_SCRIPT="$PROJECT_DIR/train.py"
DATA_DIR="$PROJECT_DIR/synthetic_data_output/dataset2/synthetic_starry_data_part_0.npz"
TUNING_DIR="$PROJECT_DIR/hpam_tuning_output_apr27"      # root for all trials
mkdir -p "$TUNING_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# 2) Fixed parameters
VAL_SPLIT=0.1
BATCH_SIZE=32
IMG_SIZE=256
# GRID_H=16
# GRID_W=16
NUM_PYRAMID=3
RES_DILATIONS="1 2 4 8"
MASK_CORNERS=true     # always true in current experiments
GPUS=1
MAX_EPOCHS=250         # keep short while sweeping; extend later on best runs

# ─────────────────────────────────────────────────────────────────────────────
# # 3) Hyper-parameter grid
# #    Comment / uncomment to narrow the search space quickly.
# LRS=(1e-4 5e-4 1e-3)
# LATENTS=(128 256)
# BASES=(32 64)
# USE_RESIDUALS=(true false)
# GRID_SIZES=(16 32)          # instead of a single value

# # Perceptual & SSIM losses
# USE_PERC=(false)            # whether to add VGG perceptual term
# LAMBDA_PERC=(0)          # weight if USE_PERC == true
# USE_SSIM=(false true)            # whether to add SSIM term
# LAMBDA_SSIM=(0.05 0.10 0.20)          # weight if USE_SSIM == true

# 3) Hyper-parameter grid
#    Comment / uncomment to narrow the search space quickly.
LRS=(5e-4 1e-3)
LATENTS=(128)
BASES=(32)
USE_RESIDUALS=(true)
GRID_SIZES=(16 32)          # instead of a single value

# Perceptual & SSIM losses
USE_PERC=(false)            # whether to add VGG perceptual term
LAMBDA_PERC=(0)          # weight if USE_PERC == true
USE_SSIM=(true)            # whether to add SSIM term
LAMBDA_SSIM=(0.10)          # weight if USE_SSIM == true


# ─────────────────────────────────────────────────────────────────────────────
# 4) Sweep all combinations (with GRID_SIZES)
for GRID in "${GRID_SIZES[@]}"; do
  GRID_H=$GRID
  GRID_W=$GRID

  for LR in "${LRS[@]}"; do
    for LAT in "${LATENTS[@]}"; do
      for BASE in "${BASES[@]}"; do
        for RES in "${USE_RESIDUALS[@]}"; do
          for PERC in "${USE_PERC[@]}"; do
            for SSIM in "${USE_SSIM[@]}"; do

              # include grid in the tag so each output dir is unique
              TAG="lr${LR}_lat${LAT}_base${BASE}_res${RES}_grid${GRID}_perc${PERC}_ssim${SSIM}"
              OUTDIR="$TUNING_DIR/$TAG"
              mkdir -p "$OUTDIR"

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
                --lr $LR
                --max_epochs $MAX_EPOCHS
                --gpus $GPUS
                --output_dir "$OUTDIR"
              )

              $RES   && CMD+=(--use_residuals)
              $MASK_CORNERS && CMD+=(--mask_corners)

              if $PERC; then
                for LP in "${LAMBDA_PERC[@]}"; do
                  PERC_TAG=${TAG}_lp${LP}
                  PERC_OUT="$TUNING_DIR/$PERC_TAG"
                  mkdir -p "$PERC_OUT"
                  PERC_CMD=("${CMD[@]}" --use_perceptual_loss --lambda_perc $LP --output_dir "$PERC_OUT")

                  if $SSIM; then
                    for LS in "${LAMBDA_SSIM[@]}"; do
                      FINAL_TAG=${PERC_TAG}_ls${LS}
                      FINAL_OUT="$TUNING_DIR/$FINAL_TAG"
                      mkdir -p "$FINAL_OUT"
                      FINAL_CMD=("${PERC_CMD[@]}" --use_ssim_loss --lambda_ssim $LS --output_dir "$FINAL_OUT")

                      echo -e "\n── Running trial: $FINAL_TAG"
                      echo "   ${FINAL_CMD[*]}"
                      "${FINAL_CMD[@]}"
                    done
                  else
                    echo -e "\n── Running trial: $PERC_TAG"
                    echo "   ${PERC_CMD[*]}"
                    "${PERC_CMD[@]}"
                  fi
                done
              else
                if $SSIM; then
                  for LS in "${LAMBDA_SSIM[@]}"; do
                    S_TAG=${TAG}_ls${LS}
                    S_OUT="$TUNING_DIR/$S_TAG"
                    mkdir -p "$S_OUT"
                    S_CMD=("${CMD[@]}" --use_ssim_loss --lambda_ssim $LS --output_dir "$S_OUT")

                    echo -e "\n── Running trial: $S_TAG"
                    echo "   ${S_CMD[*]}"
                    "${S_CMD[@]}"
                  done
                else
                  echo -e "\n── Running trial: $TAG"
                  echo "   ${CMD[*]}"
                  "${CMD[@]}"
                fi
              fi

            done
          done
        done
      done
    done
  done
done

echo -e "\nAll trials complete under different GRID sizes. Results in $TUNING_DIR/<tag>/.```
