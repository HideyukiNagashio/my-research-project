#!/bin/bash

BASE_DIR="outputs/experiments"

# 各モデルのディレクトリ
DIR1="${BASE_DIR}/transformer_X_0.5_Y_0.5_transformer_20260707_052451"
DIR2="${BASE_DIR}/hybrid_grf_X_0.5_Y_0.5_hybrid_grf_20260803_010931"
DIR3="${BASE_DIR}/hybrid_edge_X_0.5_Y_0.5_hybrid_edge_20260731_070706"

# グループラベル
LABEL1="Transformer"
LABEL2="GCN + Transformer"
LABEL3="EdgeConv + Transformer"

# 評価指標
METRICS=(
    "Fx_nrmse"
    "Fy_nrmse"
    "Fz_nrmse"
    "Fx_r2"
    "Fy_r2"
    "Fz_r2"
)

# Pythonスクリプト
SCRIPT="scripts/run_statistical_tests.py"

echo "=========================================="
echo " Statistical Tests"
echo "=========================================="
echo "Transformer     : ${DIR1}"
echo "GCN + Transformer : ${DIR2}"
echo "EdgeConv + Transformer : ${DIR3}"
echo "=========================================="

# 指定した全指標についてFriedman検定
for METRIC in "${METRICS[@]}"; do

    echo ""
    echo "##########################################"
    echo "# Metric: ${METRIC}"
    echo "##########################################"

    python "${SCRIPT}" friedman \
        --dirs "${DIR1}" "${DIR2}" "${DIR3}" \
        --labels "${LABEL1}" "${LABEL2}" "${LABEL3}" \
        --metric "${METRIC}"

done

echo ""
echo "=========================================="
echo " All statistical tests completed."
echo "=========================================="