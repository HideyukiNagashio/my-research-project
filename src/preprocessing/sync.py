import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d

def calculate_fine_offset_pressure(df_tgt, df_ref, pressure_cols, ref_col,
                                   t_start, duration=300, fs=100):
    t_end = t_start + duration
    dt = df_tgt[(df_tgt['Time (Seconds)'] >= t_start) & (df_tgt['Time (Seconds)'] <= t_end)]
    dr = df_ref[ (df_ref['Time (Seconds)']  >= t_start) & (df_ref['Time (Seconds)']  <= t_end)]
    if len(dt) < fs or len(dr) < fs:
        return 0.0
    t_min = max(dt['Time (Seconds)'].min(), dr['Time (Seconds)'].min())
    t_max = min(dt['Time (Seconds)'].max(), dr['Time (Seconds)'].max())
    if t_min >= t_max:
        return 0.0
    t_comm = np.arange(t_min, t_max, 1.0 / fs)
    def norm_sig(t_src, y):
        f = interp1d(t_src, y, kind='linear', fill_value=0, bounds_error=False)
        s = f(t_comm)
        return (s - s.mean()) / (s.std() + 1e-6)
    sig_t = norm_sig(dt['Time (Seconds)'].values, dt[pressure_cols].sum(axis=1).values)
    sig_r = norm_sig(dr['Time (Seconds)'].values, dr[ref_col].values)
    corr  = signal.correlate(sig_r, sig_t, mode='full')
    lags  = signal.correlation_lags(len(sig_r), len(sig_t), mode='full')
    return lags[np.argmax(corr)] / fs

def synchronize_merge_and_extract(df_left, df_right, df_angles, df_force, target_freq=100):
    mk1_l = df_left[df_left['Marker'] == 1]
    t_mk1 = mk1_l.iloc[0]['Time (Seconds)'] if not mk1_l.empty else 0.0
    df_right_r = df_right.copy()
    mk1_r = df_right[df_right['Marker'] == 1]
    if not mk1_r.empty:
        df_right_r['Time (Seconds)'] += t_mk1 - mk1_r.iloc[0]['Time (Seconds)']
    df_force_r = df_force.copy()
    df_force_r['Time (Seconds)'] += t_mk1
    mk2_l = df_left[df_left['Marker'] == 2]
    df_lf = df_left.copy()
    df_rf = df_right_r.copy()
    df_af = df_angles.copy()
    if not mk2_l.empty:
        t2 = mk2_l.iloc[0]['Time (Seconds)']
        pcols_l = [f'Left_Pressure_{i}'  for i in range(1, 9)]
        pcols_r = [f'Right_Pressure_{i}' for i in range(1, 9)]
        off_l = calculate_fine_offset_pressure(df_left,    df_force_r, pcols_l, 'Left_Fz',  t2)
        off_r = calculate_fine_offset_pressure(df_right_r, df_force_r, pcols_r, 'Right_Fz', t2)
        df_lf['Time (Seconds)'] += off_l
        df_af['Time (Seconds)'] += off_l
        df_rf['Time (Seconds)'] += off_r
    t_s = max(df_lf['Time (Seconds)'].min(), df_rf['Time (Seconds)'].min(),
              df_af['Time (Seconds)'].min(), df_force_r['Time (Seconds)'].min())
    t_e = min(df_lf['Time (Seconds)'].max(), df_rf['Time (Seconds)'].max(),
              df_af['Time (Seconds)'].max(), df_force_r['Time (Seconds)'].max())
    t_comm    = np.arange(t_s, t_e, 1.0 / target_freq)
    df_merged = pd.DataFrame({'Time (Seconds)': t_comm})
    for df_src in [df_lf, df_rf, df_af, df_force_r]:
        tc    = 'Time (Seconds)'
        ncols = [c for c in df_src.select_dtypes(include=[np.number]).columns
                 if c != tc and 'Marker' not in c]
        if not ncols:
            continue
        f     = interp1d(df_src[tc].values, df_src[ncols].values,
                         axis=0, kind='linear', fill_value='extrapolate')
        df_merged = pd.concat([df_merged, pd.DataFrame(f(t_comm), columns=ncols)], axis=1)
    df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]
    mk2_sync = df_lf[df_lf['Marker'] == 2]
    if not mk2_sync.empty:
        ts     = mk2_sync.iloc[0]['Time (Seconds)']
        df_out = df_merged[(df_merged['Time (Seconds)'] >= ts) &
                           (df_merged['Time (Seconds)'] <= ts + 300.0)].copy()
        df_out['Time (Seconds)'] -= ts
        return df_out.reset_index(drop=True)
    return df_merged
