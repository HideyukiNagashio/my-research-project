"""
遊脚期を含む入力 (TO-TO / HS-HS) 用の CV データセットを作成する．

既存の data/interim/raw_strides, data/processed/cv_grf は一切変更せず，別ディレクトリに出力する．

  Stage 1: 生データ -> ストライド切り出し（既存パイプラインと同一処理 + 遊脚期セグメント）
           -> data/interim/raw_strides_swing{N}/{participant}_{cond}_raw.npz
  Stage 2: 既存と同一の Group CV 分割 (run_manual_cv.GRF_FOLDS)・同一の正規化統計の算出方法で
           -> data/processed/cv_grf_swing{N}/fold*/{train,val,test}.pkl

保存するキー（pkl）:
  ensemble_X_0.5 / ensemble_Y_0.5 : 立脚期 HS->TO を 200 点に正規化（既存と同一）
  ensemble_S_pre_swing            : 遊脚期 TO(前)->HS を N 点に正規化 （TO-TO 入力の前半）
  ensemble_S_post_swing           : 遊脚期 TO->HS(次) を N 点に正規化 （HS-HS 入力の後半）
  seg_lengths                     : (n, 3) 実時間のサンプル数 [遊脚(pre), 立脚, 遊脚(post)] @100Hz

実行例:
  PYTHONPATH=. python scripts/make_swing_stance_cv.py --swing_points 140
"""
import argparse
import pickle
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

from src.preprocessing.config import (PARTICIPANTS, CONDITIONS, MASSES, ANGLE_SCALE)
from src.preprocessing.pipeline import _process_single
from src.preprocessing.normalization import compute_global_stats, apply_global_normalization
from scripts.run_manual_cv import GRF_FOLDS, load_cached_raw_data

# Stage 1 で保存する ensemble キー（compute_global_stats が X_1.0_post_swing を必要とするため含める）
RAW_KEYS = ['X_0.5', 'Y_0.5', 'X_1.0_pre_swing', 'X_1.0_post_swing', 'S_pre_swing', 'S_post_swing']
# pkl に保存するキー
PKL_KEYS = ['X_0.5', 'Y_0.5', 'S_pre_swing', 'S_post_swing']


def _worker(job):
    """1 (participant, condition) を処理して npz に保存する．戻り値は軽量なサマリのみ（メモリ節約）．"""
    participant, condition, mass, swing_points, out_dir = job
    res = _process_single((participant, condition, mass, swing_points))
    if res is None:
        return (participant, condition, None)
    path = Path(out_dir) / f"{participant}_{condition}_raw.npz"
    save = dict(columns=res['columns'], participant=participant, condition=condition,
                mass=res['mass'], durations=res['durations'], seg_lengths=res['seg_lengths'])
    for k in RAW_KEYS:
        save[f"ensemble_{k}"] = res['ensemble_dict'][k]   # 既存 raw_strides と同じ float64 のまま保存
    np.savez(path, **save)
    return (participant, condition, res['seg_lengths'].shape[0])


def run_stage1(raw_dir: Path, swing_points: int, n_workers: int):
    raw_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(p, c, MASSES[i], swing_points, str(raw_dir))
            for i, p in enumerate(PARTICIPANTS) for c in CONDITIONS
            if not (raw_dir / f"{p}_{c}_raw.npz").exists()]
    print(f"[Stage 1] {len(jobs)} trials to process (workers={n_workers}) -> {raw_dir}")
    if not jobs:
        return
    with Pool(n_workers) as pool:
        for p, c, n in pool.imap_unordered(_worker, jobs):
            print(f"  done {p}_{c}: {n} strides" if n is not None else f"  FAILED {p}_{c}")


def load_swing_raw(raw_dir: Path):
    """
    run_manual_cv.load_cached_raw_data（EXCLUDE_KEYS 適用）+ seg_lengths の読み込み．

    既存の cv_grf は process_all_data_raw の返り値順（PARTICIPANTS x CONDITIONS の順，各ブロック内は右脚→左脚）で
    結合されている（oba h,m,l -> ono h,m,l -> ...）．既存データとストライド順を一致させるため，
    同じ順序に並べ替える（Path.glob の列挙順はファイルシステム依存なので使わない）．
    """
    results = load_cached_raw_data(raw_dir, apply_exclude=True)
    results = sorted(results, key=lambda r: (PARTICIPANTS.index(r['participant']),
                                             CONDITIONS.index(r['condition'])))
    for r in results:
        with np.load(raw_dir / f"{r['participant']}_{r['condition']}_raw.npz", allow_pickle=True) as d:
            r['seg_lengths'] = d['seg_lengths']
    return results


def save_split(path: Path, results, stats, p_map, c_map):
    ids, conds, seglens = [], [], []
    ens = {k: [] for k in PKL_KEYS}
    for r in results:
        n = r['ensemble_dict']['X_1.0_post_swing'].shape[0]
        for k in PKL_KEYS:
            # 既存と同じ正規化（train のみから算出した統計を適用）．float32 保存（Dataset で float32 化されるため等価）
            ens[k].append(apply_global_normalization(r['ensemble_dict'][k].astype(np.float64), stats).astype(np.float32))
        ids.append(np.full(n, p_map[r['participant']]))
        conds.append(np.full(n, c_map[r['condition']]))
        seglens.append(r['seg_lengths'])
    out = {
        'subject_ids': np.concatenate(ids), 'condition_ids': np.concatenate(conds),
        'columns': results[0]['columns'], 'id_map': p_map, 'condition_map': c_map,
        'angle_scale': ANGLE_SCALE, 'seg_lengths': np.concatenate(seglens, axis=0),
    }
    for k in PKL_KEYS:
        out[f"ensemble_{k}"] = np.concatenate(ens[k], axis=0)
    with open(path, 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)


def run_stage2(raw_dir: Path, cv_dir: Path):
    results = load_swing_raw(raw_dir)
    if not results:
        raise RuntimeError(f"No data in {raw_dir}")
    # 既存 run_manual_cv.py と同一の ID マッピング
    unique_p = sorted(set(r['participant'] for r in results))
    p_map = {name: i for i, name in enumerate(unique_p)}
    c_map = {'h': 0, 'm': 1, 'l': 2}

    cv_dir.mkdir(parents=True, exist_ok=True)
    for fold in GRF_FOLDS:   # 既存の Group CV 分割をそのまま使用
        print(f"\n{fold['name']}  test={fold['test']}  val={fold['val']}")
        fold_dir = cv_dir / fold['name']
        fold_dir.mkdir(parents=True, exist_ok=True)
        tr = [r for r in results if r['participant'] in fold['train']]
        va = [r for r in results if r['participant'] in fold['val']]
        te = [r for r in results if r['participant'] in fold['test']]
        stats = compute_global_stats(tr, stats_path=str(fold_dir / 'normalization_stats.npz'))
        save_split(fold_dir / 'train.pkl', tr, stats, p_map, c_map)
        save_split(fold_dir / 'val.pkl', va, stats, p_map, c_map)
        save_split(fold_dir / 'test.pkl', te, stats, p_map, c_map)
        print(f"  saved -> {fold_dir}")


def main():
    ap = argparse.ArgumentParser(description="Create CV pickles for swing+stance input experiments")
    ap.add_argument('--swing_points', type=int, default=140, help='Time-normalised length of each swing segment')
    ap.add_argument('--raw_dir', type=str, default=None, help='default: data/interim/raw_strides_swing{N}')
    ap.add_argument('--cv_dir', type=str, default=None, help='default: data/processed/cv_grf_swing{N}')
    ap.add_argument('--n_workers', type=int, default=None)
    ap.add_argument('--skip_stage1', action='store_true')
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir or f"data/interim/raw_strides_swing{args.swing_points}")
    cv_dir = Path(args.cv_dir or f"data/processed/cv_grf_swing{args.swing_points}")
    if not args.skip_stage1:
        run_stage1(raw_dir, args.swing_points, args.n_workers or min(cpu_count(), 6))
    run_stage2(raw_dir, cv_dir)


if __name__ == '__main__':
    main()
