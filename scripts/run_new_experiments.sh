#!/bin/bash

echo "Starting Experiment 1: new_hybrid_edge_X_0.5_Y_0.5"
PYTHONPATH=. python scripts/train_cv.py \
    --exp_name "new_hybrid_edge_X_0.5_Y_0.5" \
    --seed 42 \
    --data_dir "data/processed/cv_grf" \
    --input_type "single_leg" \
    --target_type "grf_only" \
    --stride_type_X "0.5" \
    --stride_type_Y "0.5" \
    --model_type "hybrid_edge"

echo "Starting Experiment 2: new_hybrid_grf_X_0.5_Y_0.5"
PYTHONPATH=. python scripts/train_cv.py \
    --exp_name "new_hybrid_grf_X_0.5_Y_0.5" \
    --seed 42 \
    --data_dir "data/processed/cv_grf" \
    --input_type "single_leg" \
    --target_type "grf_only" \
    --stride_type_X "0.5" \
    --stride_type_Y "0.5" \
    --model_type "hybrid_grf"

echo "Starting Experiment 3: new_hybrid_edge_Aligned_X_0.5_Y_0.5"
PYTHONPATH=. python scripts/train_cv.py \
    --exp_name "new_hybrid_edge_Aligned_X_0.5_Y_0.5" \
    --seed 42 \
    --data_dir "data/processed/cv_grf" \
    --input_type "single_leg" \
    --target_type "grf_only" \
    --stride_type_X "0.5" \
    --stride_type_Y "0.5" \
    --model_type "hybrid_edge_aligned"

echo "Starting Experiment 4: new_hybrid_grf_Aligned_X_0.5_Y_0.5"
PYTHONPATH=. python scripts/train_cv.py \
    --exp_name "new_hybrid_grf_Aligned_X_0.5_Y_0.5" \
    --seed 42 \
    --data_dir "data/processed/cv_grf" \
    --input_type "single_leg" \
    --target_type "grf_only" \
    --stride_type_X "0.5" \
    --stride_type_Y "0.5" \
    --model_type "hybrid_grf_aligned"

echo "=========================================================="
echo " All New Experiments Completed! "
echo "=========================================================="
