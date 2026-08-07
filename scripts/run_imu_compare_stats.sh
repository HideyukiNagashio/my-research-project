#!/bin/bash

# ベースディレクトリ（実際のフォルダがあるパスに合わせて変更してください。例: outputs/experiments または outputs）
BASE_DIR="outputs/experiments" # または "outputs"

# 各モデルのディレクトリ
DIR1="${BASE_DIR}/transformer_X_0.5_Y_0.5_transformer_20260707_052451"
DIR2="${BASE_DIR}/transformer_grf_pressure_single_X_0.5_Y_0.5_transformer_20260807_045753"

# グループラベル
LABEL1="IMU ON"
LABEL2="IMU OFF"

# 評価指標
METRICS=("Fx_nrmse" "Fy_nrmse" "Fz_nrmse" "Fx_r2" "Fy_r2" "Fz_r2")

for METRIC in "${METRICS[@]}"; do
    echo "================================================================================"
    echo "Running Wilcoxon signed-rank test for metric: $METRIC"
    echo "================================================================================"
    python scripts/run_statistical_tests.py wilcoxon \
        --group1 "$DIR1" \
        --group2 "$DIR2" \
        --label1 "$LABEL1" \
        --label2 "$LABEL2" \
        --metric "$METRIC"
    echo ""
done
