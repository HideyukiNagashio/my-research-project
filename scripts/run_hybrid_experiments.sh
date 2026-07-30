#!/bin/bash

# PYTHONPATHにプロジェクトのルートを追加
export PYTHONPATH="$(pwd):$PYTHONPATH"

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

# 仮想環境のPythonを使用するパスを設定
PYTHON_EXEC="env/bin/python"

echo "=========================================================="
echo " Starting Hybrid GRF (GNN + Transformer) Training Pipeline"
echo "=========================================================="

# 実行する入力パターン(X)と，それぞれに対応する実験名(exp_name)の配列
INPUT_STRIDES=("0.5" "1.0_pre_swing" "1.0_post_swing" "1.5")
EXP_NAMES=("hybrid_grf_single_0.5" "hybrid_grf_single_pre_swing" "hybrid_grf_single_post_swing" "hybrid_grf_single_1.5")

# 配列の要素数だけループ処理
for i in "${!INPUT_STRIDES[@]}"; do
    STRIDE_X="${INPUT_STRIDES[$i]}"
    EXP_NAME="${EXP_NAMES[$i]}"

    echo "Running Experiment $((i+1)): Input X = $STRIDE_X -> Output Y = $STRIDE_Y..."
    $PYTHON_EXEC scripts/train_cv.py \
        --exp_name "$EXP_NAME" \
        --data_dir "$DATA_DIR" \
        --model_type "$MODEL" \
        --input_type "$INPUT" \
        --target_type "$TARGET" \
        --stride_type_X "$STRIDE_X" \
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

    echo "----------------------------------------------------------"
done

echo "=========================================================="
echo " All Hybrid GRF experiments completed."
echo "=========================================================="