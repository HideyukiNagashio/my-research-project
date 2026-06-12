import os
import argparse
import pickle
import numpy as np
import pandas as pd
import json
from src.training.metrics import calculate_metrics

def main():
    parser = argparse.ArgumentParser(description="Calculate subject-wise metrics for past experiments")
    parser.add_argument("exp_dir", type=str, help="Path to the past experiment directory")
    parser.add_argument("--data_dir", type=str, default="data/processed/cv", help="Path to the CV data directory")
    args = parser.parse_args()
    
    # Verify exp_dir exists
    if not os.path.exists(args.exp_dir):
        print(f"Error: Experiment directory {args.exp_dir} does not exist.")
        return
        
    # Load config to get target_type
    config_path = os.path.join(args.exp_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"Error: config.json not found in {args.exp_dir}.")
        return
        
    with open(config_path, "r") as f:
        config = json.load(f)
    target_type = config.get("target_type", "all")
    
    # Helper to get features
    def get_feature_names(target_type):
        angles = ['Hip_Flex', 'Hip_Abd', 'Hip_Rot', 'Knee_Flex', 'Knee_Abd', 'Knee_Rot', 'Ankle_Flex', 'Ankle_Abd', 'Ankle_Rot']
        grf = ['Fx', 'Fy', 'Fz']
        if target_type == 'angles_only':
            return angles
        elif target_type == 'grf_only':
            return grf
        elif target_type == 'all':
            return angles + grf
        else:
            raise ValueError(f"Unknown target_type {target_type}")
            
    all_subject_metrics = []
    
    for fold in range(1, 7):
        preds_path = os.path.join(args.exp_dir, f"preds_fold{fold}.npy")
        targets_path = os.path.join(args.exp_dir, f"targets_fold{fold}.npy")
        meta_path = os.path.join(args.exp_dir, f"sample_meta_fold{fold}.csv")
        
        if not os.path.exists(preds_path) or not os.path.exists(targets_path):
            print(f"Warning: Fold {fold} predictions/targets not found. Skipping fold {fold}.")
            continue
            
        preds = np.load(preds_path)
        targets = np.load(targets_path)
        
        # Load or reconstruct meta
        df_meta = None
        if os.path.exists(meta_path):
            df_meta = pd.read_csv(meta_path)
        else:
            # Reconstruct from test.pkl
            fold_dir = os.path.join(args.data_dir, f"fold{fold}")
            test_pkl_path = os.path.join(fold_dir, "test.pkl")
            if os.path.exists(test_pkl_path):
                print(f"Reconstructing metadata for Fold {fold} from {test_pkl_path}...")
                with open(test_pkl_path, 'rb') as f:
                    test_data = pickle.load(f)
                sub_ids = test_data.get('subject_ids')
                cond_ids = test_data.get('condition_ids')
                id_map = test_data.get('id_map', {})
                cond_map = test_data.get('condition_map', {})
                inv_id_map = {v: k for k, v in id_map.items()}
                inv_cond_map = {v: k for k, v in cond_map.items()}
                if sub_ids is not None and cond_ids is not None:
                    sub_names = [inv_id_map.get(i, "") for i in sub_ids]
                    cond_names = [inv_cond_map.get(i, "") for i in cond_ids]
                    df_meta = pd.DataFrame({
                        'sample_index': np.arange(len(sub_ids)),
                        'subject_id': sub_ids,
                        'subject_name': sub_names,
                        'condition_id': cond_ids,
                        'condition_name': cond_names
                    })
                    # Save it for convenience
                    df_meta.to_csv(meta_path, index=False)
            else:
                print(f"Error: Neither {meta_path} nor {test_pkl_path} found. Cannot evaluate Fold {fold}.")
                continue
                
        if df_meta is not None:
            unique_subjects = df_meta['subject_name'].unique()
            fold_subject_rows = []
            for sub_name in unique_subjects:
                if not sub_name:
                    continue
                mask = (df_meta['subject_name'] == sub_name).values
                sub_preds = preds[mask]
                sub_targets = targets[mask]
                
                sub_metrics = calculate_metrics(sub_targets, sub_preds)
                sub_per_feat = sub_metrics.pop('per_feature', [])
                
                row = {
                    'subject_name': sub_name,
                    'fold': fold,
                    'rmse': sub_metrics['rmse'],
                    'nrmse': sub_metrics['nrmse'],
                    'mae': sub_metrics['mae'],
                    'r2': sub_metrics['r2']
                }
                
                feature_names = get_feature_names(target_type)
                for idx, feat_name in enumerate(feature_names):
                    if idx < len(sub_per_feat):
                        row[f"{feat_name}_rmse"] = sub_per_feat[idx]['rmse']
                        row[f"{feat_name}_r2"] = sub_per_feat[idx]['r2']
                        
                fold_subject_rows.append(row)
                all_subject_metrics.append(row)
                
            if len(fold_subject_rows) > 0:
                df_sub_metrics = pd.DataFrame(fold_subject_rows)
                sub_metrics_path = os.path.join(args.exp_dir, f'subject_metrics_fold{fold}.csv')
                df_sub_metrics.to_csv(sub_metrics_path, index=False, float_format='%.4f')
                print(f"Saved subject_metrics_fold{fold}.csv")

    if len(all_subject_metrics) > 0:
        df_all_subs = pd.DataFrame(all_subject_metrics)
        df_all_subs = df_all_subs.sort_values(by=['subject_name'])
        subject_summary_path = os.path.join(args.exp_dir, 'subject_summary.csv')
        df_all_subs.to_csv(subject_summary_path, index=False, float_format='%.4f')
        print(f"Saved subject_summary.csv to {subject_summary_path}")
    else:
        print("No subject metrics were generated. Check if predictions and targets are present.")

if __name__ == "__main__":
    main()
