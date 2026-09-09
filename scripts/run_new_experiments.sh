#!/bin/bash

# モデルと実験名のペアを配列で定義
MODELS=(
    "hybrid_edge:new_hybrid_edge_X_0.5_Y_0.5"
    "hybrid_grf:new_hybrid_grf_X_0.5_Y_0.5"
    "hybrid_edge_aligned:new_hybrid_edge_Aligned_X_0.5_Y_0.5"
    "hybrid_grf_aligned:new_hybrid_grf_Aligned_X_0.5_Y_0.5"
)

echo "=========================================================="
echo " Starting 4 New Model Experiments (X=0.5, Y=0.5)"
echo "=========================================================="

for entry in "${MODELS[@]}"; do
    # コロンで分割
    MODEL="${entry%%:*}"
    EXP_NAME="${entry##*:}"
    
    echo "----------------------------------------------------------"
    echo "Running Experiment: $EXP_NAME"
    echo "Model Type = $MODEL"
    
    # 仮想環境を使用せず、コンテナ内のPythonを直接実行
    python scripts/train_cv.py \
        --exp_name "$EXP_NAME" \
        --seed 42 \
        --data_dir "data/processed/cv_grf" \
        --input_type "single_leg" \
        --target_type "grf_only" \
        --stride_type_X "0.5" \
        --stride_type_Y "0.5" \
        --model_type "$MODEL" \
        --d_model 128 \
        --nhead 4 \
        --dim_feedforward 256 \
        --hidden_dim 256 \
        --kernel_size 5 \
        --num_layers 3 \
        --dropout 0.1 \
        --gnn_out_dim 16 \
        --cnn_pool_dim 32 \
        --epochs 200 \
        --batch_size 512 \
        --lr 0.001 \
        --patience 20 \
        --factor 0.5
done

echo "=========================================================="
echo " All Experiments Completed! "
echo "=========================================================="
