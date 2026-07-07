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

def detect_toe_offs(sig, threshold=0.05, min_dist=40):
    falling = np.diff((sig > threshold).astype(int), prepend=0) == -1
    cands  = np.where(falling)[0]
    if len(cands) == 0:
        return np.array([])
    result = [cands[0]]
    for idx in cands[1:]:
        if idx - result[-1] > min_dist:
            result.append(idx)
    return np.array(result)

def slice_strides(df, fz_col, side='Left', threshold=0.05, fs=100,
                  min_dur=0.7, max_dur=1.8, apply_grf_shift=True):
    sig, time = df[fz_col].values, df['Time (Seconds)'].values
    
    hs_cands = detect_heel_strikes(sig, threshold, int(0.4 * fs))
    to_cands = detect_toe_offs(sig, threshold, int(0.4 * fs))
    
    # Store combinations in a dict
    strides_out = {
        'X_0.5': [], 'X_1.0_pre_swing': [], 'X_1.0_post_swing': [], 'X_1.5': [],
        'Y_0.5': [], 'Y_1.0_pre_swing': [], 'Y_1.0_post_swing': []
    }
    durations = []
    
    for i in range(1, len(hs_cands) - 1):
        hs_curr = hs_cands[i]
        hs_next = hs_cands[i + 1]
        
        # Find TO before hs_curr
        to_prev_list = to_cands[to_cands < hs_curr]
        if len(to_prev_list) == 0: continue
        to_prev = to_prev_list[-1]
        
        # Find TO after hs_curr but before hs_next
        to_curr_list = to_cands[(to_cands > hs_curr) & (to_cands < hs_next)]
        if len(to_curr_list) == 0: continue
        to_curr = to_curr_list[0]
        
        # Check standard 1.0 cycle duration validity
        dur = time[hs_next] - time[hs_curr]
        if not (min_dur <= dur <= max_dur):
            continue
            
        # Define the boundary indices for each pattern
        bounds = {
            'X_0.5': (hs_curr, to_curr),
            'X_1.0_pre_swing': (to_prev, to_curr),
            'X_1.0_post_swing': (hs_curr, hs_next),
            'X_1.5': (to_prev, hs_next),
            'Y_0.5': (hs_curr, to_curr),
            'Y_1.0_pre_swing': (to_prev, to_curr),
            'Y_1.0_post_swing': (hs_curr, hs_next)
        }
        
        # Bounding box for the entire stride (to cover all patterns, min to max)
        min_idx = min([b[0] for b in bounds.values()])
        max_idx = max([b[1] for b in bounds.values()])
        
        df_seg = df.iloc[min_idx:max_idx].copy()
        
        # Apply sample-wise GRF shift correction
        if apply_grf_shift:
            # The swing phases in this broad [to_prev, hs_next] interval are:
            # 1. to_prev to hs_curr
            # 2. to_curr to hs_next
            sp1_start = max(0, to_prev - min_idx)
            sp1_end = max(0, hs_curr - min_idx)
            sp2_start = max(0, to_curr - min_idx)
            sp2_end = max(0, hs_next - min_idx)
            
            grf_cols = [f"{side}_Fx", f"{side}_Fy", f"{side}_Fz", 
                        f"Contra_{'Right' if side=='Left' else 'Left'}_Fx",
                        f"Contra_{'Right' if side=='Left' else 'Left'}_Fy",
                        f"Contra_{'Right' if side=='Left' else 'Left'}_Fz"]
            
            # Since GRF offset might only be needed for the target side, we shift ipsi and contra independently.
            # Actually, just shifting the main side's GRF is safer. But let's check what's in df_seg.
            # If the noise is on the sensor side, we just shift all GRF columns.
            for s_side, s_prefix in [(side, f"{side}_"), ('Right' if side=='Left' else 'Left', "Contra_")]:
                cols = [f"{s_prefix}Fx", f"{s_prefix}Fy", f"{s_prefix}Fz"]
                cols = [c for c in cols if c in df_seg.columns] # only shift if they exist
                
                sp_data = []
                if sp1_end > sp1_start:
                    sp_data.append(df_seg.iloc[sp1_start:sp1_end][cols].values)
                if sp2_end > sp2_start:
                    sp_data.append(df_seg.iloc[sp2_start:sp2_end][cols].values)
                
                if sp_data:
                    sp_data = np.vstack(sp_data)
                    # compute mean during swing phase and subtract
                    offset = np.mean(sp_data, axis=0)
                    df_seg.loc[:, cols] -= offset

        # Slice all patterns
        for key, (start_idx, end_idx) in bounds.items():
            strides_out[key].append(df_seg.iloc[start_idx - min_idx : end_idx - min_idx].copy())
            
        durations.append(dur)
        
    print(f"  [{side}] Accepted: {len(durations)} strides")
    return strides_out, np.array(durations, dtype=np.float32)

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
