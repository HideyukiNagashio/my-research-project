#!/bin/bash

# エラーが発生した場合にスクリプトを即座に終了する
set -e

# PYTHONPATHにプロジェクトのルートを追加
export PYTHONPATH="$(pwd):$PYTHONPATH"

# システムのPythonを使用するパスを設定
PYTHON_EXEC="python"

# 共通パラメータの設定
DATA_DIR="data/processed/cv_grf"
INPUT="single_leg"
TARGET="grf_only"

# Transformer 側のパラメータ
D_MODEL=128
NHEAD=4
FF_DIM=256
LAYERS=3
DROPOUT=0.1

# GNN 側のパラメータ
GNN_OUT_DIM=16
CNN_POOL_DIM=32

# 学習パラメータ
EPOCHS=200
BATCH=512
LR=0.001
PATIENCE=20
FACTOR=0.5

# Yのストライドは固定
STRIDE_Y="0.5"

# ループで回す条件
# モデル4種類 × 入力4種類 = 16条件
MODELS=("hybrid_edge" "hybrid_gat" "hybrid_gcn_res" "hybrid_grf")
STRIDES_X=("0.5" "1.0_pre_swing" "1.0_post_swing" "1.5")

echo "========================================================================"
echo " Starting 16 Experiments for Hybrid GNN Models"
echo " Models: ${MODELS[*]}"
echo " Stride X: ${STRIDES_X[*]}"
echo " Stride Y: ${STRIDE_Y}"
echo "========================================================================"

count=1
total=16

for MODEL in "${MODELS[@]}"; do
    for X in "${STRIDES_X[@]}"; do
        
        EXP_NAME="${MODEL}_X_${X}_Y_${STRIDE_Y}"
        
        echo ""
        echo "------------------------------------------------------------------------"
        echo "Running Experiment ${count}/${total}: ${EXP_NAME}"
        echo "------------------------------------------------------------------------"
        
        $PYTHON_EXEC scripts/train_cv.py \
            --exp_name "$EXP_NAME" \
            --data_dir "$DATA_DIR" \
            --model_type "$MODEL" \
            --input_type "$INPUT" \
            --target_type "$TARGET" \
            --stride_type_X "$X" \
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
            
        count=$((count + 1))
        
    done
done

echo ""
echo "========================================================================"
echo " All 16 experiments completed successfully!"
echo "========================================================================"
