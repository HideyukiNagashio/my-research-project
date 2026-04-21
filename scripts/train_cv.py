import os
import json
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
    elif target_type == 'grf_only': out_dim = 3
    else: raise ValueError(f"Unknown target_type {target_type}")

    return in_dim, out_dim

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
        
    cv_metrics = []
    all_per_feat_metrics = []
    
    # 6-Fold クロスバリデーションの実行
    for fold in range(1, 7):
        print(f"\n{'='*40}")
        print(f" Fold {fold} / 6")
        print(f"{'='*40}")
        
        fold_dir = os.path.join(args.data_dir, f'fold{fold}')
        
        # DataLoaderの構築
        train_dataset = GaitDataset(os.path.join(fold_dir, 'train.pkl'), args.input_type, args.target_type)
        val_dataset = GaitDataset(os.path.join(fold_dir, 'val.pkl'), args.input_type, args.target_type)
        test_dataset = GaitDataset(os.path.join(fold_dir, 'test.pkl'), args.input_type, args.target_type)
        
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
            # TODO: target_typeに応じた適切な重みリストを設定するロジックが必要。
            # 今回は簡易的に、全て1.0とした上で特定の変数に重みを付けるならここを編集。
            weights = [1.0] * output_dim
            # 例: grf_only の場合は Fx, Fy, Fz なので [10.0, 10.0, 1.0] など
            if args.target_type == 'grf_only':
                weights = [10.0, 10.0, 1.0]
            elif args.target_type == 'all':
                weights = [1.0] * 9 + [10.0, 10.0, 1.0] # 角度は1.0, GRFは重みを大きく
                
            criterion = WeightedMSELoss(weights, device)
        else:
            criterion = nn.MSELoss()
            
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.factor, patience=5, verbose=True
        )
        
        # トレーナーの初期化と学習実行
        trainer = Trainer(model, criterion, optimizer, scheduler, device, patience=args.patience)
        best_wts, history = trainer.train(args.epochs, train_loader, val_loader, fold_idx=fold)
        
        # Best model の保存とロード
        model_save_path = os.path.join(exp_dir, f'best_model_fold{fold}.pth')
        torch.save(best_wts, model_save_path)
        model.load_state_dict(best_wts)
        
        # テストセットを用いた評価
        print(f"\n--- Evaluating Fold {fold} on Test Set ---")
        test_metrics, preds, targets = trainer.evaluate(test_loader)
        
        # 特徴量別のMetrics抽出および保存
        if 'per_feature' in test_metrics:
            per_feat_metrics = test_metrics.pop('per_feature')
            if len(per_feat_metrics) > 0:
                feat_df = pd.DataFrame(per_feat_metrics)
                feat_df.insert(0, 'feature', get_feature_names(args.target_type))
                feat_df.to_csv(os.path.join(exp_dir, f'feature_metrics_fold{fold}.csv'), index=False)
                
                feat_df['fold'] = fold
                all_per_feat_metrics.append(feat_df)
            
        test_metrics['fold'] = fold
        cv_metrics.append(test_metrics)
        print(f"Test RMSE: {test_metrics['rmse']:.4f} | R2: {test_metrics['r2']:.4f}")
        
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
        
    # CV全体の集計 (summary.csv の作成)
    df_metrics = pd.DataFrame(cv_metrics)
    
    # 各指標のMeanとStdを新しい行として追加
    summary_stats = df_metrics.agg(['mean', 'std'])
    summary_stats['fold'] = ['mean', 'std']
    df_metrics = pd.concat([df_metrics, summary_stats], ignore_index=True)
    
    summary_path = os.path.join(exp_dir, 'summary.csv')
    df_metrics.to_csv(summary_path, index=False)
    
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
        
        feature_summary.to_csv(os.path.join(exp_dir, 'feature_summary.csv'), index=False)

    print(f"\n{'='*40}")
    print(f"Cross Validation Complete!")
    print(f"Mean R2  : {summary_stats.loc['mean', 'r2']:.4f} ± {summary_stats.loc['std', 'r2']:.4f}")
    print(f"Mean RMSE: {summary_stats.loc['mean', 'rmse']:.4f} ± {summary_stats.loc['std', 'rmse']:.4f}")
    print(f"Results saved to: {exp_dir}")
    print(f"{'='*40}")

if __name__ == '__main__':
    main()
