#!/bin/bash

# エラーが発生した場合にスクリプトを即座に終了する
set -e

# PYTHONPATHにプロジェクトのルートを追加
export PYTHONPATH="$(pwd):$PYTHONPATH"

# 共通パラメータの設定
MODEL="transformer"
TARGET="angles_3dof"
DATA_DIR="data/processed/cv_angles"
D_MODEL=128
NHEAD=4
FF_DIM=256
HIDDEN=256
KERNEL=5
LAYERS=3
DROPOUT=0.1
EPOCHS=200
BATCH=512
LR=0.001
PATIENCE=20
FACTOR=0.5

echo "Starting 6 Transformer experiments for target: ${TARGET} (3 DoF joint angles)..."
echo "Using data directory: ${DATA_DIR}"
echo "========================================================================"

# 実験1: 片足の圧力，IMU
echo "Running Experiment 1: Single Leg (Pressure + IMU)..."
python scripts/train_cv.py \
    --exp_name trans_3dof_single_leg \
    --data_dir "$DATA_DIR" \
    --model_type "$MODEL" \
    --input_type "single_leg" \
    --target_type "$TARGET" \
    --d_model $D_MODEL \
    --nhead $NHEAD \
    --dim_feedforward $FF_DIM \
    --hidden_dim $HIDDEN \
    --kernel_size $KERNEL \
    --num_layers $LAYERS \
    --dropout $DROPOUT \
    --epochs $EPOCHS \
    --batch_size $BATCH \
    --lr $LR \
    --patience $PATIENCE \
    --factor $FACTOR

# 実験2: 片足の圧力
echo "Running Experiment 2: Pressure Single..."
python scripts/train_cv.py \
    --exp_name trans_3dof_pressure_single \
    --data_dir "$DATA_DIR" \
    --model_type "$MODEL" \
    --input_type "pressure_single" \
    --target_type "$TARGET" \
    --d_model $D_MODEL \
    --nhead $NHEAD \
    --dim_feedforward $FF_DIM \
    --hidden_dim $HIDDEN \
    --kernel_size $KERNEL \
    --num_layers $LAYERS \
    --dropout $DROPOUT \
    --epochs $EPOCHS \
    --batch_size $BATCH \
    --lr $LR \
    --patience $PATIENCE \
    --factor $FACTOR

# 実験3: 片足のIMU
echo "Running Experiment 3: IMU Single..."
python scripts/train_cv.py \
    --exp_name trans_3dof_imu_single \
    --data_dir "$DATA_DIR" \
    --model_type "$MODEL" \
    --input_type "imu_single" \
    --target_type "$TARGET" \
    --d_model $D_MODEL \
    --nhead $NHEAD \
    --dim_feedforward $FF_DIM \
    --hidden_dim $HIDDEN \
    --kernel_size $KERNEL \
    --num_layers $LAYERS \
    --dropout $DROPOUT \
    --epochs $EPOCHS \
    --batch_size $BATCH \
    --lr $LR \
    --patience $PATIENCE \
    --factor $FACTOR

# 実験4: 両足の圧力，IMU
echo "Running Experiment 4: Bilateral (Pressure + IMU)..."
python scripts/train_cv.py \
    --exp_name trans_3dof_bilateral \
    --data_dir "$DATA_DIR" \
    --model_type "$MODEL" \
    --input_type "bilateral" \
    --target_type "$TARGET" \
    --d_model $D_MODEL \
    --nhead $NHEAD \
    --dim_feedforward $FF_DIM \
    --hidden_dim $HIDDEN \
    --kernel_size $KERNEL \
    --num_layers $LAYERS \
    --dropout $DROPOUT \
    --epochs $EPOCHS \
    --batch_size $BATCH \
    --lr $LR \
    --patience $PATIENCE \
    --factor $FACTOR

# 実験5: 両足の圧力
echo "Running Experiment 5: Pressure Bilateral..."
python scripts/train_cv.py \
    --exp_name trans_3dof_pressure_bilateral \
    --data_dir "$DATA_DIR" \
    --model_type "$MODEL" \
    --input_type "pressure_bilateral" \
    --target_type "$TARGET" \
    --d_model $D_MODEL \
    --nhead $NHEAD \
    --dim_feedforward $FF_DIM \
    --hidden_dim $HIDDEN \
    --kernel_size $KERNEL \
    --num_layers $LAYERS \
    --dropout $DROPOUT \
    --epochs $EPOCHS \
    --batch_size $BATCH \
    --lr $LR \
    --patience $PATIENCE \
    --factor $FACTOR

# 実験6: 両足のIMU
echo "Running Experiment 6: IMU Bilateral..."
python scripts/train_cv.py \
    --exp_name trans_3dof_imu_bilateral \
    --data_dir "$DATA_DIR" \
    --model_type "$MODEL" \
    --input_type "imu_bilateral" \
    --target_type "$TARGET" \
    --d_model $D_MODEL \
    --nhead $NHEAD \
    --dim_feedforward $FF_DIM \
    --hidden_dim $HIDDEN \
    --kernel_size $KERNEL \
    --num_layers $LAYERS \
    --dropout $DROPOUT \
    --epochs $EPOCHS \
    --batch_size $BATCH \
    --lr $LR \
    --patience $PATIENCE \
    --factor $FACTOR

echo "All 6 experiments completed!"
