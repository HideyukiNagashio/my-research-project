"""
遊脚期・立脚期の実時間（HS/TO 検出に基づく区間長）を集計し，遊脚期の正規化点数（仮: 140）の妥当性を検討する．

入力: make_swing_stance_cv.py の Stage 1 が保存した seg_lengths（外れ値除去後の全ストライド）
      seg_lengths = [遊脚期(TO(前)->HS), 立脚期(HS->TO), 遊脚期(TO->HS(次))]  サンプル数 @100Hz
出力: 標準出力の表 と outputs/swing_stance_analysis/ 以下の CSV / 図

実行: PYTHONPATH=. python scripts/analyze_swing_stance_durations.py [--raw_dir data/interim/raw_strides_swing140]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.preprocessing.config import EXCLUDE_KEYS

FS = 100.0
N_STANCE = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_dir', type=str, default='data/interim/raw_strides_swing140')
    ap.add_argument('--out_dir', type=str, default='outputs/swing_stance_analysis')
    args = ap.parse_args()

    rows = []
    for path in sorted(Path(args.raw_dir).glob('*_raw.npz')):
        with np.load(path, allow_pickle=True) as d:
            p, c = str(d['participant']), str(d['condition'])
            if (p, c) in EXCLUDE_KEYS:
                continue
            sl = d['seg_lengths']
        df = pd.DataFrame(sl, columns=['swing_pre', 'stance', 'swing_post'])
        df.insert(0, 'condition', c)
        df.insert(0, 'participant', p)
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    df['ratio_pre'] = df['swing_pre'] / df['stance']
    df['ratio_post'] = df['swing_post'] / df['stance']
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(args.out_dir) / 'segment_lengths.csv', index=False)

    ms = lambda s: s * 1000.0 / FS
    print(f"strides: {len(df)}  (participants: {df['participant'].nunique()})")
    print("\n[Real duration, ms] (mean ± std | median | 5-95%)")
    for col in ['swing_pre', 'stance', 'swing_post']:
        v = ms(df[col])
        print(f"  {col:11s}: {v.mean():6.1f} ± {v.std():5.1f} | {v.median():6.1f} | {v.quantile(.05):6.1f} - {v.quantile(.95):6.1f}"
              f"   (min {v.min():.0f}, max {v.max():.0f})")

    print("\n[Swing/Stance duration ratio]  -> equal time-resolution swing points = 200 x ratio")
    for col in ['ratio_pre', 'ratio_post']:
        r = df[col]
        print(f"  {col:10s}: mean {r.mean():.3f}  median {r.median():.3f}  5-95% {r.quantile(.05):.3f}-{r.quantile(.95):.3f}"
              f"  => {N_STANCE * r.median():.0f} points (median), {N_STANCE * r.mean():.0f} (mean)")

    print("\n[By speed condition]  median ms (pre / stance / post) and equal-resolution points")
    g = df.groupby('condition')
    for c, sub in g:
        print(f"  {c}: n={len(sub):4d}  pre {ms(sub['swing_pre'].median()):5.0f} / stance {ms(sub['stance'].median()):5.0f} / "
              f"post {ms(sub['swing_post'].median()):5.0f} ms  -> {N_STANCE * sub['ratio_pre'].median():.0f} / "
              f"{N_STANCE * sub['ratio_post'].median():.0f} pts")

    print("\n[Points per real sample (100 Hz) after normalisation]")
    print(f"  stance (200 pts): {N_STANCE / df['stance'].median():.2f} pts/sample (median)")
    for n in (113, 140):
        print(f"  swing  ({n} pts): pre {n / df['swing_pre'].median():.2f}, post {n / df['swing_post'].median():.2f} pts/sample (median)")
    for n in (100, 120, 140, 160):
        print(f"  n_swing={n}: fraction of strides with swing points/real sample > stance's: "
              f"pre {np.mean(n / df['swing_pre'] > N_STANCE / df['stance']):.2f}, post {np.mean(n / df['swing_post'] > N_STANCE / df['stance']):.2f}")

    print("\n[Outliers] short segments (possible HS/TO detection artefacts)")
    for col, thr in [('swing_pre', 15), ('swing_post', 15), ('stance', 40)]:
        n = int((df[col] < thr).sum())
        print(f"  {col} < {thr} samples ({ms(thr):.0f} ms): {n} strides ({100 * n / len(df):.2f}%)")

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for a, col in zip(ax[:2], ['swing_pre', 'swing_post']):
        a.hist(ms(df[col]), bins=60, alpha=.7, label=col)
        a.hist(ms(df['stance']), bins=60, alpha=.4, label='stance')
        a.set_xlabel('duration [ms]'); a.legend()
    ax[2].hist(N_STANCE * df['ratio_pre'], bins=60, alpha=.6, label='pre')
    ax[2].hist(N_STANCE * df['ratio_post'], bins=60, alpha=.6, label='post')
    ax[2].axvline(140, color='k', ls='--', label='140')
    ax[2].set_xlabel('equal-resolution swing points (200 x swing/stance)'); ax[2].legend()
    plt.tight_layout()
    plt.savefig(Path(args.out_dir) / 'segment_durations.png', dpi=150)
    print(f"\nSaved: {args.out_dir}/segment_lengths.csv, segment_durations.png")


if __name__ == '__main__':
    main()

