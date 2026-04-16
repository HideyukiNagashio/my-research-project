import os
import numpy as np
from multiprocessing import Pool, cpu_count
import traceback

from .config import (PARTICIPANTS, CONDITIONS, MASSES, DEVICE_FREQ, MOCAP_FREQ, 
                     FORCE_FREQ, COLS_LEFT, COLS_RIGHT, ANGLE_SCALE)
from .loader import load_participant_data
from .cleaner import clean_mocap_columns, clean_force_columns
from .resample import process_resampling
from .filter import process_smoothing_dataframe, remove_imu_offset
from .kinematics import process_mocap_data_target_calibration
from .sync import synchronize_merge_and_extract
from .stride import slice_strides, normalize_strides_bilateral, filter_outlier_strides_mad, merge_bilateral
from .normalization import normalize_force_by_bodyweight, apply_global_normalization

def _process_single(args):
    participant, condition, mass = args
    try:
        data     = load_participant_data(participant, condition, mass)
        df_mocap = clean_mocap_columns(data['mocap'])
        df_force = clean_force_columns(data['force'])

        df_l  = process_smoothing_dataframe(process_resampling(data['left']),  fs=DEVICE_FREQ)
        df_r  = process_smoothing_dataframe(process_resampling(data['right']), fs=DEVICE_FREQ)
        df_mc = process_smoothing_dataframe(df_mocap, fs=MOCAP_FREQ)
        df_f  = process_smoothing_dataframe(df_force, fs=FORCE_FREQ)

        df_angles, _ = process_mocap_data_target_calibration(df_mc, df_l)
        df_force_bw  = normalize_force_by_bodyweight(df_f, mass)

        # ジャイロバイアス除去
        df_l_off = remove_imu_offset(df_l)
        df_r_off = remove_imu_offset(df_r)

        # 体重正規化（Phase1）
        bw = mass * 9.81
        for df_dev in [df_l_off, df_r_off]:
            pcols = [c for c in df_dev.columns if 'Pressure' in c]
            if pcols:
                df_dev[pcols] /= bw

        df_final = synchronize_merge_and_extract(df_l_off, df_r_off, df_angles, df_force_bw)

        r_strides, r_dur = slice_strides(df_final, 'Right_Fz', 'Right')
        l_strides, l_dur = slice_strides(df_final, 'Left_Fz',  'Left')

        r_dfs, r_ens = normalize_strides_bilateral(r_strides, COLS_RIGHT, COLS_LEFT)
        l_dfs, l_ens = normalize_strides_bilateral(l_strides, COLS_LEFT,  COLS_RIGHT)

        r_ens, r_dfs, _, r_dur = filter_outlier_strides_mad(r_ens, r_dfs, r_dur)
        l_ens, l_dfs, _, l_dur = filter_outlier_strides_mad(l_ens, l_dfs, l_dur)

        merged_ens, merged_cols, merged_dur = merge_bilateral(l_ens, r_ens, l_dur, r_dur)

        print(f"✓ {participant}_{condition}: {merged_ens.shape[0]} strides (52ch bilateral)")
        return dict(participant=participant, condition=condition, mass=mass,
                    ensemble=merged_ens, columns=merged_cols, durations=merged_dur)
    except Exception as e:
        print(f"✗ {participant}_{condition}: {e}")
        traceback.print_exc()
        return None

def process_all_data_raw(output_dir='data/interim/raw_strides', n_workers=None):
    os.makedirs(output_dir, exist_ok=True)
    n_workers = n_workers or min(cpu_count(), 6)
    print(f"\n【Phase 1】並列処理開始 (workers={n_workers})")
    print(f"  出力: {output_dir}  |  ensemble: (N, 200, 52ch bilateral)")

    args = [(p, c, MASSES[i])
            for i, p in enumerate(PARTICIPANTS)
            for c in CONDITIONS]

    with Pool(n_workers) as pool:
        results = pool.map(_process_single, args)

    all_results = [r for r in results if r is not None]
    print(f"\n  成功: {len(all_results)}/{len(args)} 条件")

    for r in all_results:
        path = os.path.join(output_dir, f"{r['participant']}_{r['condition']}_raw.npz")
        np.savez(path,
                 ensemble=r['ensemble'], columns=r['columns'],
                 participant=r['participant'], condition=r['condition'],
                 mass=r['mass'], durations=r['durations'])
    return all_results

def save_normalized_dataset(all_results, stats, output_dir='data/processed/normalized'):
    os.makedirs(output_dir, exist_ok=True)
    unique_p = sorted(set(r['participant'] for r in all_results))
    p_map    = {name: i for i, name in enumerate(unique_p)}
    c_map    = {'h': 0, 'm': 1, 'l': 2}

    all_ens, all_ids, all_conds, all_durs = [], [], [], []
    for r in all_results:
        norm = apply_global_normalization(r['ensemble'], stats)
        np.savez(os.path.join(output_dir, f"{r['participant']}_{r['condition']}_norm.npz"),
                 ensemble=norm, columns=r['columns'],
                 participant=r['participant'], condition=r['condition'],
                 mass=r['mass'], durations=r['durations'])
        n = norm.shape[0]
        all_ens.append(norm)
        all_ids.append(np.full(n, p_map[r['participant']]))
        all_conds.append(np.full(n, c_map[r['condition']]))
        if r['durations'] is not None:
            all_durs.append(r['durations'])

    combined = np.concatenate(all_ens, axis=0)
    combined_path = os.path.join(output_dir, 'all_data_combined.npz')
    save_kwargs = dict(
        ensemble      = combined,
        subject_ids   = np.concatenate(all_ids,   axis=0),
        condition_ids = np.concatenate(all_conds, axis=0),
        columns       = all_results[0]['columns'],
        id_map        = p_map,
        condition_map = c_map,
        angle_scale   = ANGLE_SCALE,
    )
    if all_durs:
        save_kwargs['durations'] = np.concatenate(all_durs, axis=0)
    np.savez(combined_path, **save_kwargs)

    print(f"\n  保存: {combined_path}")
    print(f"  ensemble shape: {combined.shape}  (N_strides, 200, 52)")

if __name__ == '__main__':
    import time
    from .normalization import compute_global_stats
    
    t0 = time.time()
    all_results = process_all_data_raw(output_dir='data/interim/raw_strides', n_workers=None)
    if not all_results:
        raise RuntimeError("処理できたデータがありません")

    # 注意: 全データで正規化統計量を計算する（もしCVをする場合はここを修正し、train data のリストを渡す）
    stats = compute_global_stats(all_results, stats_path='data/processed/normalization_stats.npz')

    print("\n【Phase 3 & 4】正規化の適用と保存")
    save_normalized_dataset(all_results, stats, output_dir='data/processed/normalized')

    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"  完了  (総処理時間: {elapsed/60:.1f} 分)")
    print(f"{'='*55}")
