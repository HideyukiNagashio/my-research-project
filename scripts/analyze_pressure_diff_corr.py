import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch
from src.training.dataset import GaitDataset

def load_data(data_dir):
    fold_dirs = sorted(glob.glob(os.path.join(data_dir, 'fold*')))
    
    all_x = []
    all_y = []
    
    print(f"Loading data from {len(fold_dirs)} folds in {data_dir}...")
    for fold_dir in fold_dirs:
        # train, val, testすべてを読み込んで全体で相関を見る
        for split in ['train.pkl', 'val.pkl', 'test.pkl']:
            pkl_path = os.path.join(fold_dir, split)
            if not os.path.exists(pkl_path):
                continue
                
            try:
                # GaitDatasetクラスを利用して正しくデータを抽出
                dataset = GaitDataset(
                    pkl_path, 
                    input_type='single_leg', 
                    target_type='grf_only',
                    stride_type_X='0.5', 
                    stride_type_Y='0.5'
                )
                
                # single_leg: 14次元 (前半8次元が足底圧力)
                all_x.append(dataset.X.numpy())
                # grf_only: 3次元 (Fx, Fy, Fz)
                all_y.append(dataset.y.numpy())
            except Exception as e:
                print(f"Skipping {pkl_path} due to error: {e}")
                
    if len(all_x) == 0:
        raise ValueError("データの読み込みに失敗しました。")
        
    X_concat = np.concatenate(all_x, axis=0)
    Y_concat = np.concatenate(all_y, axis=0)
    
    return X_concat, Y_concat

def main():
    data_dir = 'data/processed/cv_grf'
    try:
        X, Y = load_data(data_dir)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    print(f"Total data shape - X: {X.shape}, Y: {Y.shape}")
    
    # 対象となるエッジ (センサ間のペア)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (5, 7), (1, 6)
    ]
    
    # 前半8chが圧力
    pressure = X[:, :, :8]
    
    # Flatten (バッチと時系列をまとめる)
    P_flat = pressure.reshape(-1, 8)
    Y_flat = Y.reshape(-1, 3)
    
    targets = {
        'Fx': Y_flat[:, 0], 
        'Fy': Y_flat[:, 1], 
        'Fz': Y_flat[:, 2]
    }
    
    results = []
    
    # 各エッジにおける圧力差 (x_j - x_i) を計算し、Fx, Fy, Fzとの相関を求める
    for u, v in edges:
        # 圧力差
        diff = P_flat[:, v] - P_flat[:, u]
        
        row = {'Edge': f'({u}, {v})'}
        for t_name, t_vals in targets.items():
            # ピアソン相関係数を計算
            corr, p_value = pearsonr(diff, t_vals)
            row[f'{t_name}_Corr'] = corr
            row[f'{t_name}_p-val'] = p_value
            
        results.append(row)
        
    res_df = pd.DataFrame(results)
    
    print("\n=======================================================")
    print(" 各エッジの圧力差(j - i) と 床反力(Fx, Fy, Fz) の相関係数")
    print("=======================================================")
    # 見やすく出力
    display_df = res_df[['Edge', 'Fx_Corr', 'Fy_Corr', 'Fz_Corr']].copy()
    print(display_df.round(4).to_string(index=False))
    
    # CSVにも保存
    save_path = 'outputs/edge_pressure_diff_correlation.csv'
    os.makedirs('outputs', exist_ok=True)
    res_df.to_csv(save_path, index=False)
    print(f"\n結果を {save_path} に保存しました。")

if __name__ == '__main__':
    main()
