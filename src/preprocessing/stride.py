import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from .config import COLS_BILATERAL, COLS_IPSI, COLS_CONTRA, _FLIP_COLS

def detect_heel_strikes(sig, threshold=0.05, min_dist=40):
    rising = np.diff((sig > threshold).astype(int), prepend=0) == 1
    cands  = np.where(rising)[0]
    if len(cands) == 0:
        return np.array([])
    result = [cands[0]]
    for idx in cands[1:]:
        if idx - result[-1] > min_dist:
            result.append(idx)
    return np.array(result)

def slice_strides(df, fz_col, side='Left', threshold=0.05, fs=100,
                  min_dur=0.7, max_dur=1.8):
    sig, time = df[fz_col].values, df['Time (Seconds)'].values
    hs = detect_heel_strikes(sig, threshold, int(0.4 * fs))
    strides, durations = [], []
    for i in range(1, len(hs) - 2):
        dur = time[hs[i + 1]] - time[hs[i]]
        if min_dur <= dur <= max_dur:
            strides.append(df.iloc[hs[i]:hs[i + 1]].copy())
            durations.append(dur)
    print(f"  [{side}] Accepted: {len(strides)} strides")
    return strides, np.array(durations, dtype=np.float32)

def normalize_strides_bilateral(stride_list, ipsi_raw_cols, contra_raw_cols, n_points=200):
    x_new        = np.linspace(0, 1, n_points)
    all_raw_cols = ipsi_raw_cols + contra_raw_cols
    dfs, arrays  = [], []

    for sdf in stride_list:
        x_old = np.linspace(0, 1, len(sdf))
        mat   = np.zeros((n_points, 52))
        for j, col in enumerate(all_raw_cols):
            if col in sdf.columns:
                f = interp1d(x_old, sdf[col].values, kind='linear', fill_value='extrapolate')
                mat[:, j] = f(x_new)
        new_df = pd.DataFrame(mat, columns=COLS_BILATERAL)
        new_df.insert(0, 'Gait Cycle (%)', np.linspace(0, 100, n_points))
        dfs.append(new_df)
        arrays.append(mat)

    ens = (np.array(arrays) if arrays else np.empty((0, n_points, 52)))
    return dfs, ens

def filter_outlier_strides_mad(ensemble, stride_dfs, durations=None,
                                n_mads=3.5, ratio_thresh=0.01):
    if len(ensemble) == 0:
        dur_out = durations[:0] if durations is not None else None
        return ensemble, stride_dfs, np.array([], dtype=bool), dur_out

    median = np.median(ensemble, axis=0)
    sigma  = 1.4826 * np.median(np.abs(ensemble - median), axis=0)
    is_out = (ensemble > (median + n_mads * sigma)) | (ensemble < (median - n_mads * sigma))
    ratios = is_out.sum(axis=(1, 2)) / (ensemble.shape[1] * ensemble.shape[2])
    keep   = ratios <= ratio_thresh
    n_drop = (~keep).sum()
    if n_drop:
        print(f"  外れ値除去 (MAD): {n_drop}/{len(ensemble)} strides")

    dur_out = durations[keep] if durations is not None else None
    return ensemble[keep], [d for i, d in enumerate(stride_dfs) if keep[i]], keep, dur_out

def merge_bilateral(left_ens, right_ens, left_dur=None, right_dur=None):
    right_d = right_ens.copy()
    left_d  = left_ens.copy()

    for col_base, sign in _FLIP_COLS.items():
        if col_base in COLS_IPSI:
            left_d[:, :, COLS_IPSI.index(col_base)] *= sign
        contra_col = 'Contra_' + col_base
        if contra_col in COLS_CONTRA:
            left_d[:, :, 26 + COLS_CONTRA.index(contra_col)] *= sign

    merged = np.concatenate([right_d, left_d], axis=0)
    merged_dur = (np.concatenate([right_dur, left_dur])
                  if right_dur is not None and left_dur is not None else None)
    print(f"  R:{right_ens.shape[0]} + L(反転):{left_ens.shape[0]} "
          f"= {merged.shape[0]} strides  [52ch bilateral]")
    return merged, COLS_BILATERAL, merged_dur
