#!/bin/bash

if [[ $1 == "--help" || $1 == "-h" ]]; then
  echo "
Usage: ./run_all_pipeline.sh

Automates the full BOP dataset pipeline in the 'blenderproc' conda environment.

Steps:
  1. Activate conda environment 'blenderproc'
  2. Generate models_info.json
  3. Generate synthetic scenes
  4. Merge BOP scenes
  5. Generate SINGLESHOTPOSE labels

Edit variables at the top of this script to match your directory structure.
"
  exit 0
fi

# Set the path to your conda installation if not already in PATH
# export PATH=~/anaconda3/bin:$PATH

ENV_NAME="blenderproc"

echo "=== [Step 0] Activating conda environment: $ENV_NAME ==="
# Activate the conda environment (use 'conda' not 'source' if you installed Miniconda)
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# --- Set your variables ---
MODELS_DIR="/home/user/BlenderProc/examples/datasets/bop_challenge/output/itodd"
BOP_PARENT_PATH="/home/user/BlenderProc/examples/datasets/bop_challenge/output"
CC_TEXTURES_PATH="/home/user/cc_textures"
OUTPUT_DIR="/home/user/BlenderProc/examples/datasets/bop_challenge/output/bop_data_test/bop_data/itodd"
NUM_SCENES=10
MERGED_OUTPUT_DIR="/home/user/BlenderProc/examples/datasets/bop_challenge/output/bop_data_test/bop_data/itodd/merged"
MODEL_PATH="/home/user/BlenderProc/examples/datasets/bop_challenge/output/itodd/models/obj_000001.ply"
CAMERA_PARSER="/home/user/BlenderProc/examples/datasets/bop_challenge/output/bop_data/itodd/camera.json"
TRAIN=0.8
VAL=0.2
TEST=0.0

# --- Pipeline Steps ---
# echo "=== Step 1: Generate models_info.json ==="
# python gen_models_info.py "$MODELS_DIR" || { echo "gen_models_info.py failed"; exit 1; }

# echo "=== Step 2: Generate synthetic scenes ==="
# blenderproc run main_itodd_random_apple.py "$BOP_PARENT_PATH" "$CC_TEXTURES_PATH" "$OUTPUT_DIR" --num_scenes="$NUM_SCENES" || { echo "main_itodd_random_apple.py failed"; exit 1; }

echo "=== Step 3: Merge BOP scenes ==="
python merge_bop_scenes.py --input_root "$OUTPUT_DIR/train_pbr" --output_root "$MERGED_OUTPUT_DIR" || { echo "merge_bop_scenes.py failed"; exit 1; }

echo "=== Step 4: Generate SINGLESHOTPOSE labels ==="
python singleshotpose_labels.py --merged_dir "$MERGED_OUTPUT_DIR" --model_path "$MODEL_PATH" --camera_parser "$CAMERA_PARSER" || { echo "singleshotpose_labels.py failed"; exit 1; }

echo "=== Step 5: Generate split dataset ==="
python split_dataset.py --merged_dir "$MERGED_OUTPUT_DIR" --train_ratio "$TRAIN" --val_ratio "$VAL" --test_ratio "$TEST" || { echo "split_dataset.py failed"; exit 1; }

echo "=== All steps completed successfully! ==="
