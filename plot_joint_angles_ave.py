"""
関節角度データの可視化スクリプト
- 各条件(h, m, l)ごとに参加者全員のアンサンブル平均を重ね描きする
- カラムごとに1つの図（3条件×1列のサブプロット）
- 参加者ごとに色分け
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
#  設定
# ============================================================
DATA_DIR = "new_processed_data"

PARTICIPANTS = [
    'oba', 'ono', 'pon', 'kuno', 'john', 'konan',
    'obara', 'fukuzawa', 'kiuchi', 'yanaze', 'adachi', 'iwasaki'
]
CONDITIONS = ['h', 'm', 'l']

# プロットしたいカラム（データ内の実際のカラム名）
PLOT_COLUMNS = [
    'Hip_Flex_Ext', 'Hip_Abd_Add', 'Hip_Int_Ext_Rot',
    'Knee_Flex_Ext', 'Knee_Abd_Add', 'Knee_Int_Ext_Rot',
    'Ankle_Flex_Ext', 'Ankle_Abd_Add', 'Ankle_Int_Ext_Rot',
]

# 参加者ごとのカラーマップ
COLORS = plt.cm.tab20(np.linspace(0, 1, len(PARTICIPANTS)))

CONDITION_LABELS = {'h': 'High', 'm': 'Medium', 'l': 'Low'}

# 除外する参加者と条件の組み合わせ
EXCLUDE_KEYS = [
    ('kiuchi', 'm'),
    ('kuno', 'm'),
    ('konan', 'm'),
]

# ============================================================
#  データロード
# ============================================================
def load_data(data_dir, participants, conditions):
    """
    全参加者・全条件のデータを辞書に格納
    data[(participant, condition)] = {'ensemble': ..., 'columns': ...}
    """
    data = {}
    for p in participants:
        for c in conditions:
            file_path = os.path.join(data_dir, f"{p}_{c}_processed.npz")
            if not os.path.exists(file_path):
                print(f"  [SKIP] {file_path} が見つかりません")
                continue
            loaded = np.load(file_path)
            data[(p, c)] = {
                'ensemble': loaded['ensemble'],   # (N, 200, 26)
                'columns': list(loaded['columns']),
            }
    return data

# ============================================================
#  プロット
# ============================================================
def plot_joint_angles(data, plot_columns, participants, conditions):
    """
    カラムごとに1つの図を作成
    各図は3つのサブプロット（h, m, l 条件）
    参加者ごとに色分けしてアンサンブル平均を描画
    """
    for col_name in plot_columns:
        fig, axes = plt.subplots(1, len(conditions), figsize=(18, 5), sharey=True)
        fig.suptitle(col_name.replace("_", " "), fontsize=16, fontweight='bold')

        for ax_idx, cond in enumerate(conditions):
            ax = axes[ax_idx]
            ax.set_title(f"Condition: {CONDITION_LABELS[cond]} ({cond})", fontsize=13)
            ax.set_xlabel("Gait Cycle (%)", fontsize=11)
            if ax_idx == 0:
                ax.set_ylabel("Angle (deg)", fontsize=11)

            for p_idx, participant in enumerate(participants):
                if (participant, cond) in EXCLUDE_KEYS:
                    continue
                key = (participant, cond)
                if key not in data:
                    continue

                ensemble = data[key]['ensemble']   # (N, 200, 26)
                columns = data[key]['columns']

                if col_name not in columns:
                    
                    continue

                col_idx = columns.index(col_name)
                # 全歩行周期のデータ: (N, 200)
                all_cycles = ensemble[:, :, col_idx]
                # アンサンブル平均
                if all_cycles.size == 0:
                    print(f"DEBUG: Data for {participant} is empty")
                    continue

                mean_curve = np.mean(all_cycles, axis=0)
                x = np.linspace(0, 100, mean_curve.shape[0])
                ax.plot(x, mean_curve, color=COLORS[p_idx], label=participant,
                        linewidth=1.5, alpha=0.85)

            ax.grid(True, alpha=0.3)

        # 凡例は最後のサブプロットの右側に配置
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', fontsize=9,
                   bbox_to_anchor=(1.08, 0.5), title="Participant")

        plt.tight_layout(rect=[0, 0, 0.92, 0.95])

        # 保存
        save_path = f"ave_R-plots_joint_{col_name}.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  保存: {save_path}")

    plt.show()


# ============================================================
#  実行
# ============================================================
if __name__ == "__main__":
    print("\n=== プロット処理開始 ===")
    data = load_data(DATA_DIR, PARTICIPANTS, CONDITIONS)
    plot_joint_angles(data, PLOT_COLUMNS, PARTICIPANTS, CONDITIONS)