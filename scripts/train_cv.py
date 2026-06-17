import os
import json
import pickle
import time
import datetime
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.utils import seed_everything
from src.training.config import get_args_parser
from src.training.dataset import GaitDataset
from src.training.loss import WeightedMSELoss
from src.training.engine import Trainer
from src.models import get_model
from src.training.metrics import calculate_metrics

def get_dimensions(input_type, target_type):
    # Calculate input_dim
    if input_type == 'single_leg': in_dim = 14
    elif input_type == 'bilateral': in_dim = 28
    elif input_type == 'pressure_single': in_dim = 8
    elif input_type == 'pressure_bilateral': in_dim = 16
    elif input_type == 'imu_single': in_dim = 6
    elif input_type == 'imu_bilateral': in_dim = 12
    else: raise ValueError(f"Unknown input_type {input_type}")

    # Calculate output_dim
    if target_type == 'all': out_dim = 12
    elif target_type == 'angles_only': out_dim = 9
    elif target_type == 'angles_6dof': out_dim = 6
    elif target_type == 'angles_3dof': out_dim = 3
    elif target_type == 'grf_only': out_dim = 3
    else: raise ValueError(f"Unknown target_type {target_type}")

    return in_dim, out_dim

def get_input_feature_names(input_type):
    ipsi_press = [f"P{i}" for i in range(1, 9)]
    ipsi_imu = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"]
    ipsi = ipsi_press + ipsi_imu
    
    contra_press = [f"Contra_P{i}" for i in range(1, 9)]
    contra_imu = ["Contra_AccX", "Contra_AccY", "Contra_AccZ", "Contra_GyroX", "Contra_GyroY", "Contra_GyroZ"]
    contra = contra_press + contra_imu
    
    if input_type == 'single_leg': return ipsi
    elif input_type == 'bilateral': return ipsi + contra
    elif input_type == 'pressure_single': return ipsi_press
    elif input_type == 'pressure_bilateral': return ipsi_press + contra_press
    elif input_type == 'imu_single': return ipsi_imu
    elif input_type == 'imu_bilateral': return ipsi_imu + contra_imu
    else: return []
def get_feature_names(target_type):
    angles = ['Hip_Flex', 'Hip_Abd', 'Hip_Rot', 'Knee_Flex', 'Knee_Abd', 'Knee_Rot', 'Ankle_Flex', 'Ankle_Abd', 'Ankle_Rot']
    grf = ['Fx', 'Fy', 'Fz']
    if target_type == 'angles_only':
        return angles
    elif target_type == 'angles_6dof':
        return ['Hip_Flex', 'Hip_Abd', 'Hip_Rot', 'Knee_Flex', 'Ankle_Flex', 'Ankle_Abd']
    elif target_type == 'angles_3dof':
        return ['Hip_Flex', 'Knee_Flex', 'Ankle_Flex']
    elif target_type == 'grf_only':
        return grf
    elif target_type == 'all':
        return angles + grf
    else:
        raise ValueError(f"Unknown target_type {target_type}")

def main():
    parser = get_args_parser()
    args = parser.parse_args()
    
    # 実行環境のセットアップ
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # モデルの入出力次元を決定
    input_dim, output_dim = get_dimensions(args.input_type, args.target_type)
    
    # 出力先ディレクトリの作成
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join('outputs', 'experiments', f"{args.exp_name}_{args.model_type}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # 設定の保存
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    try:
        with open(os.path.join(exp_dir, 'feature_names.json'), 'w') as f:
            json.dump(get_input_feature_names(args.input_type), f, indent=4)
        with open(os.path.join(exp_dir, 'target_names.json'), 'w') as f:
            json.dump(get_feature_names(args.target_type), f, indent=4)
        print("Saved feature_names.json and target_names.json")
    except Exception as e:
        print(f"Warning: could not save feature/target names: {e}")
        
    cv_metrics = []
    all_per_feat_metrics = []
    all_subject_metrics = []
    
    # Detect the number of folds dynamically based on fold directories
    import glob
    fold_dirs = sorted(glob.glob(os.path.join(args.data_dir, 'fold*')))
    num_folds = len(fold_dirs)
    if num_folds == 0:
        raise FileNotFoundError(f"No fold directories (e.g. fold1, fold2, ...) found in {args.data_dir}")
    print(f"Detected {num_folds} folds in {args.data_dir}")
    
    # クロスバリデーションの実行
    for fold in range(1, num_folds + 1):
        print(f"\n{'='*40}")
        print(f" Fold {fold} / {num_folds}")
        print(f"{'='*40}")
        
        fold_dir = os.path.join(args.data_dir, f'fold{fold}')
        
        # DataLoaderの構築
        train_dataset = GaitDataset(os.path.join(fold_dir, 'train.pkl'), args.input_type, args.target_type)
        val_dataset = GaitDataset(os.path.join(fold_dir, 'val.pkl'), args.input_type, args.target_type)
        test_dataset = GaitDataset(os.path.join(fold_dir, 'test.pkl'), args.input_type, args.target_type)
        
        y_mean_np = None
        y_std_np = None
        if args.standardize_y:
            y_flat = train_dataset.y.view(-1, output_dim)
            y_mean = torch.mean(y_flat, dim=0)
            y_std = torch.std(y_flat, dim=0)
            y_std = torch.clamp(y_std, min=1e-8)
            
            train_dataset.y = (train_dataset.y - y_mean) / y_std
            val_dataset.y = (val_dataset.y - y_mean) / y_std
            test_dataset.y = (test_dataset.y - y_mean) / y_std
            
            y_mean_np = y_mean.cpu().numpy()
            y_std_np = y_std.cpu().numpy()
            
            try:
                stats_save_path = os.path.join(exp_dir, f'y_standardization_stats_fold{fold}.json')
                with open(stats_save_path, 'w') as f:
                    json.dump({
                        'fold': fold,
                        'mean': y_mean_np.tolist(),
                        'std': y_std_np.tolist()
                    }, f, indent=4)
                print(f"Applying Target Standardization (Z-score):")
                print(f"  Means: {['{:.4f}'.format(m) for m in y_mean_np.tolist()]}")
                print(f"  Stds:  {['{:.4f}'.format(s) for s in y_std_np.tolist()]}")
                print(f"  Saved standardization stats to {stats_save_path}")
            except Exception as e:
                print(f"  Warning: could not save standardization stats: {e}")
                
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # モデルの構築
        model_kwargs = {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'dropout_prob': args.dropout
        }
        if args.model_type == 'cnn':
            model_kwargs.update({'hidden_dim': args.hidden_dim, 'kernel_size': args.kernel_size})
        elif args.model_type == 'bilstm':
            model_kwargs.update({'hidden_dim': args.hidden_dim, 'num_layers': args.num_layers})
        elif args.model_type == 'transformer':
            model_kwargs.update({'d_model': args.d_model, 'nhead': args.nhead, 
                                 'num_layers': args.num_layers, 'dim_feedforward': args.dim_feedforward})
            
        model = get_model(args.model_type, **model_kwargs).to(device)
        
        # Loss関数
        if args.use_weighted_loss:
            if args.weight_type == 'variance':
                # 分散の逆数による重み付け
                y_flat = train_dataset.y.view(-1, output_dim)
                variances = torch.var(y_flat, dim=0)
                variances = torch.clamp(variances, min=1e-8)
                weights = (1.0 / variances).tolist()
            elif args.weight_type == 'std':
                # 標準偏差の逆数による重み付け
                y_flat = train_dataset.y.view(-1, output_dim)
                stds = torch.std(y_flat, dim=0)
                stds = torch.clamp(stds, min=1e-8)
                weights = (1.0 / stds).tolist()
            else:  # 'fixed'
                weights = [1.0] * output_dim
                if args.target_type == 'grf_only':
                    weights = [10.0, 10.0, 1.0]
                elif args.target_type == 'all':
                    weights = [1.0] * 9 + [10.0, 10.0, 1.0]
            
            print(f"Applying Weighted Loss (weight_type={args.weight_type}):")
            print(f"  Weights: {['{:.4f}'.format(w) for w in weights]}")
            
            # 後で確認できるように重みパラメータをJSONとして保存
            try:
                weights_save_path = os.path.join(exp_dir, f'loss_weights_fold{fold}.json')
                with open(weights_save_path, 'w') as f:
                    json.dump({
                        'fold': fold,
                        'weight_type': args.weight_type,
                        'target_type': args.target_type,
                        'weights': weights
                    }, f, indent=4)
                print(f"  Saved weights config to {weights_save_path}")
            except Exception as e:
                print(f"  Warning: could not save weights config: {e}")
                
            criterion = WeightedMSELoss(weights, device)
        else:
            criterion = nn.MSELoss()
            
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.factor, patience=5
        )
        
        # トレーナーの初期化と学習実行
        trainer = Trainer(
            model, criterion, optimizer, scheduler, device, 
            patience=args.patience, y_mean=y_mean_np, y_std=y_std_np
        )
        best_wts, history = trainer.train(args.epochs, train_loader, val_loader, fold_idx=fold)
        
        # Best model の保存とロード
        model_save_path = os.path.join(exp_dir, f'best_model_fold{fold}.pth')
        torch.save(best_wts, model_save_path)
        model.load_state_dict(best_wts)
        
        # テストセットを用いた評価
        print(f"\n--- Evaluating Fold {fold} on Test Set ---")
        test_metrics, preds, targets = trainer.evaluate(test_loader)
        
        # --- 予測、ターゲット、入力のアレイ保存 ---
        try:
            all_inputs = []
            for batch_x, batch_y in test_loader:
                all_inputs.append(batch_x.cpu().numpy())
            inputs_array = np.concatenate(all_inputs, axis=0)
            
            np.save(os.path.join(exp_dir, f'inputs_fold{fold}.npy'), inputs_array)
            np.save(os.path.join(exp_dir, f'preds_fold{fold}.npy'), preds)
            np.save(os.path.join(exp_dir, f'targets_fold{fold}.npy'), targets)
        except Exception as e:
            print(f"Warning: could not save npy arrays: {e}")
            
        # --- メタデータの保存 ---
        try:
            with open(os.path.join(fold_dir, 'test.pkl'), 'rb') as f:
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
                meta_path = os.path.join(exp_dir, f'sample_meta_fold{fold}.csv')
                df_meta.to_csv(meta_path, index=False)
                
                # 被験者ごとの評価指標を計算
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
                    
                    feature_names = get_feature_names(args.target_type)
                    for idx, feat_name in enumerate(feature_names):
                        if idx < len(sub_per_feat):
                            row[f"{feat_name}_nrmse"] = sub_per_feat[idx]['nrmse']
                            row[f"{feat_name}_r2"] = sub_per_feat[idx]['r2']
                            
                    fold_subject_rows.append(row)
                    all_subject_metrics.append(row)
                    
                if len(fold_subject_rows) > 0:
                    df_sub_metrics = pd.DataFrame(fold_subject_rows)
                    sub_metrics_path = os.path.join(exp_dir, f'subject_metrics_fold{fold}.csv')
                    df_sub_metrics.to_csv(sub_metrics_path, index=False, float_format='%.4f')
        except Exception as e:
            print(f"Warning: could not save sample_meta_fold{fold}.csv: {e}")
            
        print("Saved:")
        print(f"  - preds_fold{fold}.npy")
        print(f"  - targets_fold{fold}.npy")
        print(f"  - inputs_fold{fold}.npy")
        print(f"  - sample_meta_fold{fold}.csv")
        if 'fold_subject_rows' in locals() and len(fold_subject_rows) > 0:
            print(f"  - subject_metrics_fold{fold}.csv")
        
        # 特徴量別のMetrics抽出および保存
        if 'per_feature' in test_metrics:
            per_feat_metrics = test_metrics.pop('per_feature')
            if len(per_feat_metrics) > 0:
                feat_df = pd.DataFrame(per_feat_metrics)
                feat_df.insert(0, 'feature', get_feature_names(args.target_type))
                feat_df.to_csv(os.path.join(exp_dir, f'feature_metrics_fold{fold}.csv'), index=False, float_format='%.4f')
                
                print(f"\n--- Fold {fold} Feature Metrics ---")
                print(feat_df.round(4).to_string(index=False))
                
                feat_df['fold'] = fold
                all_per_feat_metrics.append(feat_df)
            
        test_metrics['fold'] = fold
        cv_metrics.append(test_metrics)
        
        # 学習曲線の保存
        plt.figure(figsize=(8, 6))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title(f'Fold {fold} Training Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(exp_dir, f'learning_curve_fold{fold}.png'))
        plt.close()
    
    # === feature_summary.csv の作成 ===
    if len(all_per_feat_metrics) > 0:
        all_feat_df = pd.concat(all_per_feat_metrics, ignore_index=True)
        # 集計用に使った fold 列は消す
        all_feat_df.drop(columns=['fold'], inplace=True, errors='ignore')
        
        # featureでグループ化して平均と標準偏差を計算
        feature_summary = all_feat_df.groupby('feature', sort=False).agg(['mean', 'std'])
        # MultiIndexの列名（RMSEのMeanなど）を結合
        feature_summary.columns = [f"{col[0]}_{col[1]}" for col in feature_summary.columns]
        feature_summary.reset_index(inplace=True)
        
        # 元の特徴量順序通りにソートする
        ordered_features = get_feature_names(args.target_type)
        feature_summary['feature'] = pd.Categorical(feature_summary['feature'], categories=ordered_features, ordered=True)
        feature_summary = feature_summary.sort_values('feature')
        
        feature_summary.to_csv(os.path.join(exp_dir, 'feature_summary.csv'), index=False, float_format='%.4f')

    # === subject_summary.csv の作成 ===
    if len(all_subject_metrics) > 0:
        df_all_subs = pd.DataFrame(all_subject_metrics)
        df_all_subs = df_all_subs.sort_values(by=['subject_name'])
        subject_summary_path = os.path.join(exp_dir, 'subject_summary.csv')
        df_all_subs.to_csv(subject_summary_path, index=False, float_format='%.4f')
        print(f"Saved subject_summary.csv to {subject_summary_path}")
        
    print(f"\n{'='*40}")
    print("Cross Validation Complete!")
    print("\n--- Feature Summary (Mean ± Std) ---")
    if len(all_per_feat_metrics) > 0:
        print(feature_summary.round(4).to_string(index=False))
    print(f"\nResults saved to: {exp_dir}")
    print(f"{'='*40}")

if __name__ == '__main__':
    main()
