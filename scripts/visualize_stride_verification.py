import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_experiment_strides(exp_dir):
    exp_path = Path(exp_dir)
    config_file = exp_path / "config.json"
    inputs_file = exp_path / "inputs_fold1.npy"
    targets_file = exp_path / "targets_fold1.npy"
    
    if not (config_file.exists() and inputs_file.exists() and targets_file.exists()):
        print(f"必要なファイルが {exp_dir} に見つかりません。")
        print(f"（学習が完了していないか、パスが間違っています）")
        return

    with open(config_file, 'r') as f:
        config = json.load(f)
        
    target_type = config.get("target_type", "grf_only")
    config_x = config.get("stride_type_X", "unknown")
    config_y = config.get("stride_type_Y", "unknown")
    
    print(f"データをロード中: {exp_dir}")
    inputs = np.load(inputs_file)
    targets = np.load(targets_file)
    
    # 最初の8ch（同側足底圧力）の合計の平均波形
    mean_input_pressure = np.mean(np.sum(inputs[:, :, 0:8], axis=2), axis=0)
    
    # ターゲットのFzの平均波形
    if target_type == 'grf_only':
        fz_idx = 2
    elif target_type == 'all':
        fz_idx = 11
    else:
        fz_idx = -1
        
    if fz_idx != -1:
        mean_target_fz = np.mean(targets[:, :, fz_idx], axis=0)
    else:
        mean_target_fz = np.mean(np.sum(np.abs(targets), axis=2), axis=0)
        
    # プロットの作成
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    # Input Plot
    axs[0].plot(mean_input_pressure, label='Mean Total Pressure (ipsi)', color='blue', linewidth=2)
    axs[0].set_title(f"Input X (Config: {config_x})")
    axs[0].set_xlabel("Time step (0-200)")
    axs[0].set_ylabel("Pressure Sum")
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].legend()
    
    # Target Plot
    axs[1].plot(mean_target_fz, label='Mean Fz', color='red', linewidth=2)
    axs[1].set_title(f"Target Y (Config: {config_y})")
    axs[1].set_xlabel("Time step (0-200)")
    axs[1].set_ylabel("Force (BW)")
    axs[1].grid(True, linestyle='--', alpha=0.7)
    axs[1].legend()
    
    plt.tight_layout()
    out_img = exp_path / "stride_verification.png"
    plt.savefig(out_img, dpi=150)
    plt.close()
    
    print(f"可視化画像を保存しました: {out_img}")
    print("この画像を開いて、波形が意図したストライドパターン（スイング期がどこにあるか）になっているか目視確認してください。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True, help="Path to experiment directory (e.g. outputs/experiments/hybrid_edge_X_0.5_Y_0.5...)")
    args = parser.parse_args()
    visualize_experiment_strides(args.exp_dir)
