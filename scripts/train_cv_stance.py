"""
遊脚期情報を含む入力 (TO-TO / HS-HS) と従来の HS-TO を，同一の Transformer 設定・同一の Group CV で比較する．

  HS-TO : 入力 200x14               -> Encoder 200x128           -> 立脚期 200x128 -> 200x3
  TO-TO : 入力 (n_swing+200)x14     -> Encoder (n_swing+200)x128 -> [n_swing:]     -> 200x3
  HS-HS : 入力 (200+n_swing)x14     -> Encoder (200+n_swing)x128 -> [:200]         -> 200x3

既存の scripts/train_cv.py は変更せず，学習ループ・評価・保存形式のみ同等のものをここに実装する．
学習条件の既定値は基準実験 (outputs/experiments/transformer_X_0.5_Y_0.5_transformer_*/config.json) に合わせてある:
  d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1, MSE, Adam(lr=1e-3),
  ReduceLROnPlateau(factor=0.5, patience=5), epochs=200, batch_size=512, early stopping patience=20, seed=42

実行例:
  PYTHONPATH=. python scripts/train_cv_stance.py --condition TO-TO --exp_name swing140
  PYTHONPATH=. python scripts/train_cv_stance.py --condition HS-HS --folds 1 --epochs 2   # 動作確認
"""
import os
import json
import pickle
import time
import datetime
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.utils import seed_everything
from src.training.dataset_swing import GaitSwingStanceDataset, get_window_spec, CONDITIONS
from src.training.engine import Trainer
from src.training.metrics import calculate_metrics
from src.models import get_model
from scripts.train_cv import get_dimensions, get_input_feature_names, get_feature_names


def get_args():
    p = argparse.ArgumentParser(description='6-Fold Group CV: swing+stance input, stance-only output')
    p.add_argument('--exp_name', type=str, default='swing_stance')
    p.add_argument('--condition', type=str, required=True, choices=list(CONDITIONS))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--seed_mode', type=str, default='per_fold', choices=['per_fold', 'once'],
                   help="per_fold: 各 Fold 開始時に同じ seed で初期化（条件間で初期重みが一致）. "
                        "once: 従来の train_cv.py と同様に最初に1回だけ")
    p.add_argument('--data_dir', type=str, default=None, help='default: data/processed/cv_grf_swing{n_swing}')
    p.add_argument('--n_swing', type=int, default=140, help='遊脚期の時間正規化点数（データ作成時の値と一致させる）')
    p.add_argument('--pe_mode', type=str, default='aligned', choices=['aligned', 'sequential'],
                   help="aligned: 立脚期の位置インデックスを全条件で 0..199 に揃える / sequential: 先頭を0とする従来方式")
    p.add_argument('--input_type', type=str, default='single_leg', choices=['single_leg'])
    p.add_argument('--target_type', type=str, default='grf_only', choices=['grf_only'])
    p.add_argument('--folds', type=int, nargs='+', default=None, help='実行する Fold 番号 (default: 全Fold)')
    # model
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--nhead', type=int, default=4)
    p.add_argument('--num_layers', type=int, default=3)
    p.add_argument('--dim_feedforward', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.1)
    # training
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--factor', type=float, default=0.5)
    return p.parse_args()


def main():
    args = get_args()
    if args.data_dir is None:
        args.data_dir = f'data/processed/cv_grf_swing{args.n_swing}'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    seed_everything(args.seed)

    input_dim, output_dim = get_dimensions(args.input_type, args.target_type)
    spec = get_window_spec(args.condition, n_swing=args.n_swing, pe_mode=args.pe_mode)
    print(f"Condition {args.condition}: {spec}")

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join('outputs', 'experiments', f"{args.exp_name}_{args.condition}_transformer_stance_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump({**vars(args), 'model_type': 'transformer_stance', 'window_spec': spec.to_dict()}, f, indent=4)
    with open(os.path.join(exp_dir, 'feature_names.json'), 'w') as f:
        json.dump(get_input_feature_names(args.input_type), f, indent=4)
    with open(os.path.join(exp_dir, 'target_names.json'), 'w') as f:
        json.dump(get_feature_names(args.target_type), f, indent=4)

    import glob
    num_folds = len(sorted(glob.glob(os.path.join(args.data_dir, 'fold*'))))
    if num_folds == 0:
        raise FileNotFoundError(f"No fold directories found in {args.data_dir}")
    folds = args.folds or list(range(1, num_folds + 1))
    print(f"Detected {num_folds} folds in {args.data_dir}; running folds {folds}")

    cv_metrics, all_per_feat, all_subject_metrics = [], [], []
    feature_names = get_feature_names(args.target_type)

    for fold in folds:
        print(f"\n{'=' * 40}\n Fold {fold} / {num_folds}  [{args.condition}]\n{'=' * 40}")
        if args.seed_mode == 'per_fold':
            seed_everything(args.seed)
        fold_dir = os.path.join(args.data_dir, f'fold{fold}')

        mk = lambda s: GaitSwingStanceDataset(os.path.join(fold_dir, f'{s}.pkl'), args.condition,
                                              args.input_type, args.target_type,
                                              n_swing=args.n_swing, pe_mode=args.pe_mode)
        train_ds, val_ds, test_ds = mk('train'), mk('val'), mk('test')
        print(f"  X: {tuple(train_ds.X.shape[1:])}  y: {tuple(train_ds.y.shape[1:])}  "
              f"(train/val/test = {len(train_ds)}/{len(val_ds)}/{len(test_ds)})")

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        model = get_model('transformer_stance', input_dim=input_dim, output_dim=output_dim,
                          dropout_prob=args.dropout, d_model=args.d_model, nhead=args.nhead,
                          num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
                          seq_len=spec.seq_len, stance_start=spec.stance_start,
                          stance_len=spec.stance_len, pos_start=spec.pos_start).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.factor,
                                                         patience=5, threshold=1e-4)
        trainer = Trainer(model, criterion, optimizer, scheduler, device, patience=args.patience)
        best_wts, history = trainer.train(args.epochs, train_loader, val_loader, fold_idx=fold)

        torch.save(best_wts, os.path.join(exp_dir, f'best_model_fold{fold}.pth'))
        model.load_state_dict(best_wts)

        print(f"\n--- Evaluating Fold {fold} on Test Set ---")
        test_metrics, preds, targets = trainer.evaluate(test_loader)
        assert preds.shape == targets.shape == (len(test_ds), spec.stance_len, output_dim), preds.shape
        assert np.isfinite(preds).all(), "NaN/Inf in predictions"

        np.save(os.path.join(exp_dir, f'inputs_fold{fold}.npy'), test_ds.X.numpy())
        np.save(os.path.join(exp_dir, f'preds_fold{fold}.npy'), preds)
        np.save(os.path.join(exp_dir, f'targets_fold{fold}.npy'), targets)

        # --- メタデータ・被験者別指標（train_cv.py と同じ形式） ---
        with open(os.path.join(fold_dir, 'test.pkl'), 'rb') as f:
            test_data = pickle.load(f)
        inv_id = {v: k for k, v in test_data.get('id_map', {}).items()}
        inv_cond = {v: k for k, v in test_data.get('condition_map', {}).items()}
        sub_ids, cond_ids = test_data['subject_ids'], test_data['condition_ids']
        df_meta = pd.DataFrame({
            'sample_index': np.arange(len(sub_ids)), 'subject_id': sub_ids,
            'subject_name': [inv_id.get(i, '') for i in sub_ids],
            'condition_id': cond_ids, 'condition_name': [inv_cond.get(i, '') for i in cond_ids]})
        if 'seg_lengths' in test_data:
            df_meta[['len_swing_pre', 'len_stance', 'len_swing_post']] = test_data['seg_lengths']
        df_meta.to_csv(os.path.join(exp_dir, f'sample_meta_fold{fold}.csv'), index=False)

        rows = []
        for sub in df_meta['subject_name'].unique():
            m = (df_meta['subject_name'] == sub).values
            sm = calculate_metrics(targets[m], preds[m])
            per = sm.pop('per_feature', [])
            row = {'subject_name': sub, 'fold': fold, 'rmse': sm['rmse'], 'nrmse': sm['nrmse'],
                   'mae': sm['mae'], 'r2': sm['r2']}
            for i, fn in enumerate(feature_names):
                row[f"{fn}_nrmse"], row[f"{fn}_r2"] = per[i]['nrmse'], per[i]['r2']
            rows.append(row)
            all_subject_metrics.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(exp_dir, f'subject_metrics_fold{fold}.csv'),
                                  index=False, float_format='%.4f')

        per_feat = test_metrics.pop('per_feature')
        feat_df = pd.DataFrame(per_feat)
        feat_df.insert(0, 'feature', feature_names)
        feat_df.to_csv(os.path.join(exp_dir, f'feature_metrics_fold{fold}.csv'), index=False, float_format='%.4f')
        print(f"\n--- Fold {fold} Feature Metrics ---\n{feat_df.round(4).to_string(index=False)}")
        feat_df['fold'] = fold
        all_per_feat.append(feat_df)

        test_metrics['fold'] = fold
        test_metrics['n_epochs_run'] = len(history['train_loss'])
        cv_metrics.append(test_metrics)

        plt.figure(figsize=(8, 6))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title(f'Fold {fold} Training Curve ({args.condition})')
        plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(exp_dir, f'learning_curve_fold{fold}.png'))
        plt.close()

    # --- 集計 ---
    df_cv = pd.DataFrame(cv_metrics)
    mean_row = df_cv.mean(numeric_only=True); mean_row['fold'] = 'mean'
    std_row = df_cv.std(numeric_only=True); std_row['fold'] = 'std'
    pd.concat([df_cv, pd.DataFrame([mean_row, std_row])], ignore_index=True).to_csv(
        os.path.join(exp_dir, 'summary.csv'), index=False, float_format='%.4f')

    all_feat_df = pd.concat(all_per_feat, ignore_index=True).drop(columns=['fold'])
    fs = all_feat_df.groupby('feature', sort=False).agg(['mean', 'std'])
    fs.columns = [f"{c[0]}_{c[1]}" for c in fs.columns]
    fs.reset_index().to_csv(os.path.join(exp_dir, 'feature_summary.csv'), index=False, float_format='%.4f')
    pd.DataFrame(all_subject_metrics).sort_values('subject_name').to_csv(
        os.path.join(exp_dir, 'subject_summary.csv'), index=False, float_format='%.4f')

    print(f"\n{'=' * 40}\nCross Validation Complete! [{args.condition}]")
    print(pd.concat([df_cv, pd.DataFrame([mean_row, std_row])]).round(4).to_string(index=False))
    print(f"\nResults saved to: {exp_dir}\n{'=' * 40}")


if __name__ == '__main__':
    main()

