"""
生歩行データ (Raw Strides) の可視化スクリプト
- 各条件(h, m, l)ごとに参加者全員のアンサンブル平均を描画する
- カラムごとに1つの図（3条件×1列のサブプロット）
- 参加者ごとに色分け
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# ============================================================
#  設定
# ============================================================
DATA_DIR = "data/interim/raw_strides"
OUTPUT_DIR = "outputs/raw_strides_plots"

# 実験協力者のリスト (raw_strides ディレクトリ内のファイルから抽出)
PARTICIPANTS = [
    'adachi', 'fukuzawa', 'iwasaki', 'john', 'kiuchi', 'konan',
    'kuno', 'oba', 'obara', 'ono', 'pon', 'yanaze'
]
CONDITIONS = ['h', 'm', 'l']
CONDITION_LABELS = {'h': 'High', 'm': 'Medium', 'l': 'Low'}

# 参加者ごとのカラーマップ
COLORS = plt.cm.tab20(np.linspace(0, 1, len(PARTICIPANTS)))

# # 除外する参加者と条件の組み合わせ
EXCLUDE_KEYS = [
#     ('kiuchi', 'm'),
#     ('kuno', 'm'),
#     ('konan', 'm'),
]

# ============================================================
#  処理
# ============================================================
def load_data():
    """
    全参加者・全条件のデータを辞書に格納
    data[(participant, condition)] = {'ensemble': ..., 'columns': ...}
    さらに全てで共通のカラム名リストを取得
    """
    data = {}
    all_columns = []
    
    for p in PARTICIPANTS:
        for c in CONDITIONS:
            file_path = os.path.join(DATA_DIR, f"{p}_{c}_raw.npz")
            if not os.path.exists(file_path):
                print(f"  [SKIP] {file_path} が見つかりません")
                continue
            
            try:
                loaded = np.load(file_path, allow_pickle=True)
                ensemble = loaded['ensemble'] # (strides, sequence_length, columns)
                cols = list(loaded['columns'])
                
                # ★ 追加: durations が保存されていれば読み込む（古いファイルへの配慮）
                durations = loaded['durations'] if 'durations' in loaded else np.array([])
                
                data[(p, c)] = {
                    'ensemble': ensemble,
                    'columns': cols,
                    'durations': durations  # ★ 追加
                }
                if not all_columns:
                    all_columns = cols
            except Exception as e:
                print(f"  [ERROR] {file_path} の読み込みに失敗しました: {e}")
                
    return data, all_columns

def plot_raw_strides(data, plot_columns):
    """
    カラムごとに1つの図を作成
    各図は3つのサブプロット（h, m, l 条件）
    参加者ごとに色分けしてアンサンブル平均を描画
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for col_name in plot_columns:
        fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(18, 5), sharey=True)
        fig.suptitle(col_name.replace("_", " "), fontsize=16, fontweight='bold')

        for ax_idx, cond in enumerate(CONDITIONS):
            ax = axes[ax_idx]
            ax.set_title(f"Condition: {CONDITION_LABELS[cond]} ({cond})", fontsize=13)
            ax.set_xlabel("Gait Cycle (%)", fontsize=11)
            if ax_idx == 0:
                ax.set_ylabel("Value", fontsize=11)

            for p_idx, participant in enumerate(PARTICIPANTS):
                # 除外リストに含まれていたらスキップ
                if (participant, cond) in EXCLUDE_KEYS:
                    continue

                key = (participant, cond)
                if key not in data:
                    continue

                ensemble = data[key]['ensemble']
                columns = data[key]['columns']

                if col_name not in columns:
                    continue

                col_idx = columns.index(col_name)
                # 全歩行周期のデータ: (strides, sequence_length)
                all_cycles = ensemble[:, :, col_idx]

                if all_cycles.size == 0 or all_cycles.shape[0] == 0:
                    continue

                mean_curve = np.mean(all_cycles, axis=0)
                x = np.linspace(0, 100, mean_curve.shape[0])
                ax.plot(x, mean_curve, color=COLORS[p_idx], label=participant,
                        linewidth=1.5, alpha=0.85)
                        
            ax.grid(True, alpha=0.3)

        # 凡例は最初のサブプロットから作成して図の右側に配置
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='center right', fontsize=9,
                       bbox_to_anchor=(1.08, 0.5), title="Participant")

        plt.tight_layout(rect=[0, 0, 0.92, 0.95])

        # 保存
        safe_col_name = col_name.replace("/", "_").replace(" ", "_")
        save_path = os.path.join(OUTPUT_DIR, f"plot_{safe_col_name}_ave.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  保存: {save_path}")
        plt.close(fig)

def plot_stride_durations(data):
    """
    参加者・条件ごとの歩行所要時間（スカラー値）をエラーバー付き棒グラフで可視化
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(18, 5), sharey=True)
    fig.suptitle("Stride Durations (Seconds)", fontsize=16, fontweight='bold')

    for ax_idx, cond in enumerate(CONDITIONS):
        ax = axes[ax_idx]
        ax.set_title(f"Condition: {CONDITION_LABELS[cond]} ({cond})", fontsize=13)
        ax.set_xlabel("Participant", fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel("Duration [s]", fontsize=11)

        means = []
        stds = []
        x_labels = []
        bar_colors = []

        for p_idx, participant in enumerate(PARTICIPANTS):
            if (participant, cond) in EXCLUDE_KEYS:
                continue

            key = (participant, cond)
            if key not in data or 'durations' not in data[key]:
                continue

            durs = data[key]['durations']
            if len(durs) == 0:
                continue

            # 平均と標準偏差を計算
            means.append(np.mean(durs))
            stds.append(np.std(durs))
            x_labels.append(participant)
            bar_colors.append(COLORS[p_idx])

        # 棒グラフの描画
        if means:
            x_pos = np.arange(len(x_labels))
            ax.bar(x_pos, means, yerr=stds, color=bar_colors, alpha=0.8, capsize=4, edgecolor='black')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
            
        ax.grid(True, alpha=0.3, axis='y') # Y軸のみグリッドを表示

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 保存
    save_path = os.path.join(OUTPUT_DIR, "plot_stride_durations_ave.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  保存: {save_path}")
    plt.close(fig)

# ============================================================
#  実行
# ============================================================
if __name__ == "__main__":
    print("\n=== 生歩行データ プロット処理開始 ===")
    data_dict, cols = load_data()
    if cols:
        print(f"見つかったカラム数: {len(cols)}")
        plot_raw_strides(data_dict, cols)
        
        # ★ 追加: 所要時間のプロットを実行
        plot_stride_durations(data_dict)
        
        print("=== 終了 ===")
    else:
        print("有効なデータが見つかりませんでした。")
