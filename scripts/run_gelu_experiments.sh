#!/bin/bash

# エラーが発生した場合にスクリプトを即座に終了する
set -e

# PYTHONPATHにプロジェクトのルートを追加
export PYTHONPATH="$(pwd):$PYTHONPATH"

# 仮想環境のPythonを使用するパスを設定
PYTHON_EXEC="env/bin/python"

# 共通パラメータの設定
MODEL="transformer_gelu"
TARGET="grf_only"
INPUT="single_leg"
DATA_DIR="data/processed/cv_grf"
D_MODEL=128
NHEAD=4
FF_DIM=256
LAYERS=3
DROPOUT=0.1
EPOCHS=200
BATCH=512
LR=0.001
PATIENCE=20
FACTOR=0.5
STRIDE_Y="0.5"

echo "Starting 4 Transformer GeLU experiments with Output Stride (Y): ${STRIDE_Y}..."
echo "========================================================================"

# 実験1: Input X = 0.5
echo "Running Experiment 1: Input X = 0.5 -> Output Y = 0.5..."
$PYTHON_EXEC scripts/train_cv.py \
    --exp_name "transformer_X_0.5_Y_0.5" \
    --data_dir "$DATA_DIR" \
    --model_type "$MODEL" \
    --input_type "$INPUT" \
    --target_type "$TARGET" \
    --stride_type_X "0.5" \
    --stride_type_Y "$STRIDE_Y" \
    --d_model $D_MODEL \
    --nhead $NHEAD \
    --dim_feedforward $FF_DIM \
    --num_layers $LAYERS \
    --dropout $DROPOUT \
    --epochs $EPOCHS \
    --batch_size $BATCH \
    --lr $LR \
    --patience $PATIENCE \
    --factor $FACTOR

# 実験2: Input X = 1.0_pre_swing
echo "Running Experiment 2: Input X = 1.0_pre_swing -> Output Y = 0.5..."
$PYTHON_EXEC scripts/train_cv.py \
    --exp_name "transformer_X_1.0_pre_swing_Y_0.5" \
    --data_dir "$DATA_DIR" \
    --model_type "$MODEL" \
    --input_type "$INPUT" \
    --target_type "$TARGET" \
    --stride_type_X "1.0_pre_swing" \
    --stride_type_Y "$STRIDE_Y" \
    --d_model $D_MODEL \
    --nhead $NHEAD \
    --dim_feedforward $FF_DIM \
    --num_layers $LAYERS \
    --dropout $DROPOUT \
    --epochs $EPOCHS \
    --batch_size $BATCH \
    --lr $LR \
    --patience $PATIENCE \
    --factor $FACTOR

# 実験3: Input X = 1.0_post_swing
echo "Running Experiment 3: Input X = 1.0_post_swing -> Output Y = 0.5..."
$PYTHON_EXEC scripts/train_cv.py \
    --exp_name "transformer_X_1.0_post_swing_Y_0.5" \
    --data_dir "$DATA_DIR" \
    --model_type "$MODEL" \
    --input_type "$INPUT" \
    --target_type "$TARGET" \
    --stride_type_X "1.0_post_swing" \
    --stride_type_Y "$STRIDE_Y" \
    --d_model $D_MODEL \
    --nhead $NHEAD \
    --dim_feedforward $FF_DIM \
    --num_layers $LAYERS \
    --dropout $DROPOUT \
    --epochs $EPOCHS \
    --batch_size $BATCH \
    --lr $LR \
    --patience $PATIENCE \
    --factor $FACTOR

# 実験4: Input X = 1.5
echo "Running Experiment 4: Input X = 1.5 -> Output Y = 0.5..."
$PYTHON_EXEC scripts/train_cv.py \
    --exp_name "transformer_X_1.5_Y_0.5" \
    --data_dir "$DATA_DIR" \
    --model_type "$MODEL" \
    --input_type "$INPUT" \
    --target_type "$TARGET" \
    --stride_type_X "1.5" \
    --stride_type_Y "$STRIDE_Y" \
    --d_model $D_MODEL \
    --nhead $NHEAD \
    --dim_feedforward $FF_DIM \
    --num_layers $LAYERS \
    --dropout $DROPOUT \
    --epochs $EPOCHS \
    --batch_size $BATCH \
    --lr $LR \
    --patience $PATIENCE \
    --factor $FACTOR

echo "All 4 experiments completed!"
