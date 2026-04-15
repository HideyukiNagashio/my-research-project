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
                data[(p, c)] = {
                    'ensemble': ensemble,
                    'columns': cols,
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

                x = np.linspace(0, 100, all_cycles.shape[1])

                for i, cycle in enumerate(all_cycles):
                    if i == 0:
                        ax.plot(x, cycle,
                                color=COLORS[p_idx],
                                linewidth=0.8,
                                alpha=0.2,
                                label=participant)
                    else:
                        ax.plot(x, cycle,
                                color=COLORS[p_idx],
                                linewidth=0.8,
                                alpha=0.2)

            ax.grid(True, alpha=0.3)

        # 凡例は最初のサブプロットから作成して図の右側に配置
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='center right', fontsize=9,
                       bbox_to_anchor=(1.08, 0.5), title="Participant")

        plt.tight_layout(rect=[0, 0, 0.92, 0.95])

        # 保存
        safe_col_name = col_name.replace("/", "_").replace(" ", "_")
        save_path = os.path.join(OUTPUT_DIR, f"plot_{safe_col_name}.png")
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
        print("=== 終了 ===")
    else:
        print("有効なデータが見つかりませんでした。")
