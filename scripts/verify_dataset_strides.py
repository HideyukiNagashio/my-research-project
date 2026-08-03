import os
import json
import numpy as np
from pathlib import Path

def classify_stride_pattern(mean_signal):
    """
    波形（例：Fzの平均波形）から、どのストライドパターンでスライスされたかを分類する。
    - 0.5: 始点から終点まで立脚期（ゼロ期間がほぼない）
    - 1.0_pre_swing: 最初がスイング期（ゼロ）、後半が立脚期
    - 1.0_post_swing: 最初が立脚期、後半がスイング期（ゼロ）
    - 1.5: 最初と最後の両方にスイング期（ゼロ）がある
    """
    max_val = np.max(mean_signal)
    if max_val == 0:
        return "unknown (all zero)"
        
    threshold = 0.1 * max_val  # 最大値の10%をゼロ判定の閾値とする
    
    # 全200サンプルのうち、最初の20%（0〜40）と最後の20%（160〜200）の平均値をチェック
    start_mean = np.mean(mean_signal[:40])
    end_mean = np.mean(mean_signal[160:])
    
    start_zero = start_mean < threshold
    end_zero = end_mean < threshold
    
    if start_zero and end_zero:
        return "1.5"
    elif start_zero and not end_zero:
        return "1.0_pre_swing"
    elif not start_zero and end_zero:
        return "1.0_post_swing"
    else:
        return "0.5"

def verify_experiments(base_dir="outputs/experiments"):
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"ディレクトリが見つかりません: {base_dir}")
        return

    exp_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    found_any = False
    
    print("="*80)
    print(" 過去の実験データのストライドパターン検証")
    print("="*80)
    
    for exp_dir in sorted(exp_dirs):
        config_file = exp_dir / "config.json"
        inputs_file = exp_dir / "inputs_fold1.npy"
        targets_file = exp_dir / "targets_fold1.npy"
        
        # .npyファイルが存在する実験のみを対象とする
        if not (config_file.exists() and inputs_file.exists() and targets_file.exists()):
            continue
            
        found_any = True
        
        # 1. Configから想定されるストライド設定を取得
        with open(config_file, 'r') as f:
            config = json.load(f)
            
        config_x = config.get("stride_type_X", "unknown")
        config_y = config.get("stride_type_Y", "unknown")
        target_type = config.get("target_type", "grf_only")
        
        # 2. .npyデータから実際の波形を読み込む
        inputs = np.load(inputs_file)
        targets = np.load(targets_file)
        
        # 3. 入力データ（X）の判定
        # 最初の8チャンネル（同側の足底圧力センサ）の合計を使用
        mean_input_pressure = np.mean(np.sum(inputs[:, :, 0:8], axis=2), axis=0)
        actual_x = classify_stride_pattern(mean_input_pressure)
        
        # 4. 出力データ（Y）の判定
        # target_type に応じて Fz のインデックスを決定
        if target_type == 'grf_only':
            fz_idx = 2
        elif target_type == 'all':
            fz_idx = 11  # angles 9ch + Fx, Fy, Fz
        else:
            fz_idx = -1 # 角度のみの場合は判定が難しいため全体の絶対値和を使用
            
        if fz_idx != -1:
            mean_target_fz = np.mean(targets[:, :, fz_idx], axis=0)
        else:
            mean_target_fz = np.mean(np.sum(np.abs(targets), axis=2), axis=0)
            
        actual_y = classify_stride_pattern(mean_target_fz)
        
        # 5. 結果の判定と表示
        match_x = "✓" if actual_x == config_x else "✗ MISMATCH"
        match_y = "✓" if actual_y == config_y else "✗ MISMATCH"
        
        print(f"\n実験名: {exp_dir.name}")
        print(f"  入力 (X) - Config設定: {config_x:15} | 実際のデータ: {actual_x:15} [{match_x}]")
        print(f"  出力 (Y) - Config設定: {config_y:15} | 実際のデータ: {actual_y:15} [{match_y}]")
        
    if not found_any:
        print("\n検証可能な .npy ファイルを含む完了済みの実験データが見つかりませんでした。")
        print("（学習が途中でエラーになった、または完了していない可能性があります）")
        print("="*80)

if __name__ == "__main__":
    verify_experiments()
