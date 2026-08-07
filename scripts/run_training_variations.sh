#!/bin/bash

# --- 共通パラメータ ---
COMMON_ARGS=(
    --seed 42
    --data_dir "data/processed/cv"
    --input_type "pressure_single"
    --target_type "grf_only"
    --model_type "transformer"
    --d_model 128
    --nhead 4
    --dim_feedforward 256
    --hidden_dim 256
    --kernel_size 5
    --num_layers 3
    --dropout 0.1
    --epochs 200
    --batch_size 512
    --lr 0.001
    --patience 20
    --factor 0.5
    --stride_type_Y "0.5"
    # use_weighted_loss は「なし」のため付与しません
)

BASE_EXP_NAME="transformer_grf_pressure_single"

echo "========================================================="
echo " Training 1: Input(X) = 0.5, Target(Y) = 0.5"
echo "========================================================="
EXP_NAME="${BASE_EXP_NAME}_X_0.5_Y_0.5"
python scripts/train_cv.py "${COMMON_ARGS[@]}" --exp_name "$EXP_NAME" --stride_type_X "0.5"

echo "========================================================="
echo " Training 2: Input(X) = pre_swing, Target(Y) = 0.5"
echo "========================================================="
EXP_NAME="${BASE_EXP_NAME}_X_1.0_pre_swing_Y_0.5"
python scripts/train_cv.py "${COMMON_ARGS[@]}" --exp_name "$EXP_NAME" --stride_type_X "1.0_pre_swing"

echo "========================================================="
echo " Training 3: Input(X) = post_swing, Target(Y) = 0.5"
echo "========================================================="
EXP_NAME="${BASE_EXP_NAME}_X_1.0_post_swing_Y_0.5"
python scripts/train_cv.py "${COMMON_ARGS[@]}" --exp_name "$EXP_NAME" --stride_type_X "1.0_post_swing"
