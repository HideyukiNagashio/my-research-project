#!/bin/bash

# ベースディレクトリ（実際のフォルダがあるパスに合わせて変更してください。例: outputs）
BASE_DIR="outputs"

# 各モデルのディレクトリ
DIR1="${BASE_DIR}/transformer_X_0.5_Y_0.5_transformer_20260707_052451"
DIR2="${BASE_DIR}/transformer_X_1.0_post_swing_Y_0.5_transformer_20260707_054836"
DIR3="${BASE_DIR}/transformer_X_1.0_pre_swing_Y_0.5_transformer_20260707_053726"
DIR4="${BASE_DIR}/transformer_X_1.5_Y_0.5_transformer_20260707_060238"

# グループラベル
LABELS=("0.5 to 0.5" "post to 0.5" "pre to 0.5" "1.5 to 0.5")

# 評価指標
METRICS=("Fx_nrmse" "Fy_nrmse" "Fz_nrmse" "Fx_r2" "Fy_r2" "Fz_r2")

for METRIC in "${METRICS[@]}"; do
    echo "================================================================================"
    echo "Running Friedman test for metric: $METRIC"
    echo "================================================================================"
    python scripts/run_statistical_tests.py friedman \
        --dirs "$DIR1" "$DIR2" "$DIR3" "$DIR4" \
        --labels "${LABELS[@]}" \
        --metric "$METRIC"
    echo ""
done
