#!/bin/bash

# Configuration settings matching the transformer_grf_single configuration
DATA_DIR="data/processed/cv"
MODEL="hybrid_grf"
INPUT="single_leg"
TARGET="grf_only"

# Transformer architecture settings
D_MODEL=128
NHEAD=4
FF_DIM=256
LAYERS=3
DROPOUT=0.1

# GNN specific architecture settings
GNN_OUT_DIM=16
CNN_POOL_DIM=32

# Training settings
EPOCHS=200
BATCH=512
LR=0.001
PATIENCE=20
FACTOR=0.5
STRIDE_Y="0.5" # From previous experiments

PYTHON_EXEC="python"

echo "=========================================================="
echo " Starting Hybrid GRF (GNN + Transformer) Training Pipeline"
echo "=========================================================="

echo "Running Experiment 1: Input X = 1.0_pre_swing -> Output Y = 0.5..."
$PYTHON_EXEC scripts/train_cv.py \
    --exp_name "hybrid_grf_single" \
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
    --gnn_out_dim $GNN_OUT_DIM \
    --cnn_pool_dim $CNN_POOL_DIM \
    --epochs $EPOCHS \
    --batch_size $BATCH \
    --lr $LR \
    --patience $PATIENCE \
    --factor $FACTOR

# 実験2: Input X = 1.0_post_swing (if needed)
echo "Running Experiment 2: Input X = 1.0_post_swing -> Output Y = 0.5..."
$PYTHON_EXEC scripts/train_cv.py \
    --exp_name "hybrid_grf_single_post_swing" \
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
    --gnn_out_dim $GNN_OUT_DIM \
    --cnn_pool_dim $CNN_POOL_DIM \
    --epochs $EPOCHS \
    --batch_size $BATCH \
    --lr $LR \
    --patience $PATIENCE \
    --factor $FACTOR

echo "=========================================================="
echo " All Hybrid GRF experiments completed."
echo "=========================================================="
