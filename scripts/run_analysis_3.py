import os
import sys
import json
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 14,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})

# Add project root to path so we can import src
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from src.models import get_model
from src.training.dataset import GaitDataset
from src.training.engine import Trainer
from src.training.metrics import calculate_metrics

BASE_DIR = os.path.join(repo_root, "outputs/experiments")
EDGE_EXP_DIR = os.path.join(BASE_DIR, "new_hybrid_edge_X_0.5_Y_0.5_hybrid_edge_20261006_070416")
OUTPUT_DIR = os.path.join(repo_root, "outputs/analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)
TARGET_FEATURES = ["Fx", "Fy", "Fz"]

def get_dimensions(input_type, target_type):
    if input_type == 'single_leg': in_dim = 14
    elif input_type == 'bilateral': in_dim = 28
    elif input_type == 'pressure_single': in_dim = 8
    elif input_type == 'pressure_bilateral': in_dim = 16
    elif input_type == 'imu_single': in_dim = 6
    elif input_type == 'imu_bilateral': in_dim = 12
    else: in_dim = 16

    if target_type == 'all': out_dim = 12
    elif target_type == 'angles_only': out_dim = 9
    elif target_type == 'angles_6dof': out_dim = 6
    elif target_type == 'angles_3dof': out_dim = 3
    elif target_type == 'grf_only': out_dim = 3
    else: out_dim = 3
    return in_dim, out_dim

def main():
    print("--- Starting Analysis 3: Edge Ablation ---")
    
    config_path = os.path.join(EDGE_EXP_DIR, "config.json")
    if not os.path.exists(config_path):
        print(f"Config not found at {config_path}")
        return
        
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    input_dim, output_dim = get_dimensions(config.get('input_type', 'pressure_bilateral'), 
                                           config.get('target_type', 'grf_only'))
                                           
    model_kwargs = {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'dropout_prob': config.get('dropout', 0.1),
        'd_model': config.get('d_model', 128),
        'nhead': config.get('nhead', 4),
        'num_layers': config.get('num_layers', 3),
        'dim_feedforward': config.get('dim_feedforward', 256),
        'use_shortcut': config.get('use_shortcut', True),
        'gnn_out_dim': config.get('gnn_out_dim', 16),
        'cnn_pool_dim': config.get('cnn_pool_dim', 32)
    }
    
    # 1. Discover edges to ablate from the first fold's checkpoint
    fold1_ckpt_path = os.path.join(EDGE_EXP_DIR, 'best_model_fold1.pth')
    state_dict_fold1 = torch.load(fold1_ckpt_path, map_location=device)
    orig_edge_index = state_dict_fold1['edge_index'].clone()
    
    edges_to_ablate = []
    for i in range(orig_edge_index.shape[1]):
        u = orig_edge_index[0, i].item()
        v = orig_edge_index[1, i].item()
        if u < v:
            if (u, v) not in edges_to_ablate:
                edges_to_ablate.append((u, v))
                
    print(f"Discovered {len(edges_to_ablate)} unique edges to ablate from checkpoint.")
    
    subject_ablation_results = []
    
    fold_files = glob.glob(os.path.join(EDGE_EXP_DIR, "best_model_fold*.pth"))
    num_folds = len(fold_files)
    
    if num_folds == 0:
        print("No best_model_fold*.pth found in experiment directory.")
        return
        
    for fold in range(1, num_folds + 1):
        print(f"--- Processing Fold {fold} ---")
        model = get_model(config['model_type'], **model_kwargs).to(device)
        
        state_dict = torch.load(os.path.join(EDGE_EXP_DIR, f'best_model_fold{fold}.pth'), map_location=device)
        
        # Override buffers to match the checkpoint shape to avoid size mismatch
        if 'edge_index' in state_dict:
            model.edge_index = state_dict['edge_index'].clone().to(device)
            del state_dict['edge_index']
        if 'norm_coords' in state_dict:
            model.norm_coords = state_dict['norm_coords'].clone().to(device)
            del state_dict['norm_coords']
            
        model.load_state_dict(state_dict, strict=False)
        
        stats_path = os.path.join(EDGE_EXP_DIR, f'y_standardization_stats_fold{fold}.json')
        y_mean_np, y_std_np = None, None
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                y_mean_np = np.array(stats['mean'])
                y_std_np = np.array(stats['std'])
                
        data_dir = config.get('data_dir', 'data/processed/cv_grf')
        # Handle case where data_dir in config might be relative to repo root
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(repo_root, data_dir)
            
        fold_dir = os.path.join(data_dir, f'fold{fold}')
        test_dataset = GaitDataset(
            os.path.join(fold_dir, 'test.pkl'),
            config.get('input_type', 'pressure_bilateral'),
            config.get('target_type', 'grf_only'),
            config.get('stride_type_X', '1.0'),
            config.get('stride_type_Y', '1.0')
        )
        
        test_loader = DataLoader(test_dataset, batch_size=config.get('batch_size', 64), shuffle=False)
        
        criterion = torch.nn.MSELoss()
        trainer = Trainer(model, criterion, None, None, device, patience=1, y_mean=y_mean_np, y_std=y_std_np)
        
        # Baseline Evaluation
        print("  Evaluating baseline...")
        model.edge_index = orig_edge_index.to(device)
        _, preds_base, targets_base = trainer.evaluate(test_loader)
        
        meta = pd.read_csv(os.path.join(EDGE_EXP_DIR, f'sample_meta_fold{fold}.csv'))
        
        # Ablation Evaluations
        ablation_preds = {}
        for (u, v) in edges_to_ablate:
            print(f"  Evaluating without edge P{u+1}-P{v+1}...")
            # Remove (u,v) and (v,u)
            mask = ~(((orig_edge_index[0] == u) & (orig_edge_index[1] == v)) | 
                     ((orig_edge_index[0] == v) & (orig_edge_index[1] == u)))
            model.edge_index = orig_edge_index[:, mask].to(device)
            
            _, preds_abl, _ = trainer.evaluate(test_loader)
            ablation_preds[(u, v)] = preds_abl
            
        # Subject-wise metric calculation
        unique_subjects = meta["subject_name"].unique()
        for sub in unique_subjects:
            mask_sub = (meta["subject_name"] == sub).values
            targets_sub = targets_base[mask_sub]
            
            base_metrics = calculate_metrics(targets_sub, preds_base[mask_sub])
            
            for (u, v) in edges_to_ablate:
                abl_metrics = calculate_metrics(targets_sub, ablation_preds[(u, v)][mask_sub])
                
                row = {
                    "subject": sub,
                    "fold": fold,
                    "edge": f"P{u+1}-P{v+1}"
                }
                
                for i, feat in enumerate(TARGET_FEATURES):
                    b_nrmse = base_metrics['per_feature'][i]['nrmse']
                    a_nrmse = abl_metrics['per_feature'][i]['nrmse']
                    
                    row[f"baseline_{feat}_NRMSE"] = b_nrmse
                    row[f"ablated_{feat}_NRMSE"] = a_nrmse
                    row[f"delta_{feat}_NRMSE"] = a_nrmse - b_nrmse
                    
                subject_ablation_results.append(row)
                
    df_subject_ablation = pd.DataFrame(subject_ablation_results)
    df_subject_ablation.to_csv(os.path.join(OUTPUT_DIR, "edge_ablation_subjectwise.csv"), index=False)
    
    # Aggregate over all subjects
    df_ablation = df_subject_ablation.drop(columns=["subject", "fold"], errors='ignore').groupby("edge", as_index=False).mean()
    
    # Sort by delta_Fx_NRMSE
    df_ablation = df_ablation.sort_values(by="delta_Fx_NRMSE", ascending=False)
    df_ablation.to_csv(os.path.join(OUTPUT_DIR, "edge_ablation_results.csv"), index=False)
    
    # Plotting Fx Ablation
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_ablation, x="edge", y="delta_Fx_NRMSE", order=df_ablation["edge"], color="tomato")
    plt.axhline(0, color='black', linestyle='--')
    plt.ylabel("Delta NRMSE")
    plt.xlabel("Removed Edge")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_analysis3_edge_ablation_bar.png"), dpi=300)
    plt.close()
    
    print("Analysis 3 completed. Results saved to", OUTPUT_DIR)

if __name__ == "__main__":
    main()
