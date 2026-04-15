"""
歩行データ前処理スクリプト（高速化版）

【ipynbからの主な変更点】
  正規化:
    IMU      → 静止区間オフセット除去（条件ごと） + グローバルstdスケーリング
    足底圧力 → グローバルmin-max（全データ or trainのみ）
    関節角度 → /180 固定定数
    床反力   → 体重正規化のまま（変更なし）
  外れ値除去:
    std → MAD（頑健推定）

【高速化のポイント】
  1. 関節角度計算: Pythonループ → NumPy完全ベクトル化（約120x faster）
  2. 補間処理:     列ごとloop  → axis=0 一括補間（約3x faster）
  3. 外部ループ:   逐次処理    → multiprocessing.Pool で並列化

【処理フロー】
  Phase 1: 全データの生ストライド抽出（正規化前）
  Phase 2: グローバル統計量の計算（全データ or trainのみで実行）
  Phase 3: 正規化適用 → 保存
"""

import os
import warnings
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy import signal
from numpy import linalg as LA
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')

# ============================================================
# 実験設定
# ============================================================
PARTICIPANTS = ['oba', 'ono', 'pon', 'kuno', 'john', 'konan',
                'obara', 'fukuzawa', 'kiuchi', 'yanaze', 'adachi', 'iwasaki']
MASSES       = [60.9, 63.8, 68.7, 65.9, 77, 74, 63.8, 64.5, 73.2, 53.5, 70.7, 47.9]
CONDITIONS   = ['h', 'm', 'l']

DEVICE_FREQ  = 100
MOCAP_FREQ   = 250
FORCE_FREQ   = 1000
CUTOFF_FREQ  = 6
FILTER_ORDER = 4
ANGLE_SCALE  = 180.0   # 関節角度スケーリング定数（逆変換: pred × 180）

# マーカーインデックス
RI  = 0; RGT = 1; RK = 2; RM = 3; RT = 4
LI  = 5; LGT = 6; LK = 7; LM = 8; LT = 9
JOINT_NO = 6

SIGN_TABLE = {
    'hip':   np.array([ 1.,  1.,  1.]),
    'knee':  np.array([-1.,  1.,  1.]),
    'ankle': np.array([ 1.,  1.,  1.]),
}

COLS_LEFT = [
    'Left_Pressure_1', 'Left_Pressure_2', 'Left_Pressure_3', 'Left_Pressure_4',
    'Left_Pressure_5', 'Left_Pressure_6', 'Left_Pressure_7', 'Left_Pressure_8',
    'Left_Accel_X', 'Left_Accel_Y', 'Left_Accel_Z',
    'Left_Gyro_X',  'Left_Gyro_Y',  'Left_Gyro_Z',
    'Left_Hip_Flex_Ext',   'Left_Hip_Abd_Add',   'Left_Hip_Int_Ext_Rot',
    'Left_Knee_Flex_Ext',  'Left_Knee_Abd_Add',  'Left_Knee_Int_Ext_Rot',
    'Left_Ankle_Flex_Ext', 'Left_Ankle_Abd_Add', 'Left_Ankle_Int_Ext_Rot',
    'Left_Fx', 'Left_Fy', 'Left_Fz',
]
COLS_RIGHT = [c.replace('Left', 'Right') for c in COLS_LEFT]

# ============================================================
# 1. データ読み込み
# ============================================================
def load_participant_data(participant, condition, mass):
    print(f"\n{'='*55}")
    print(f"  {participant} - {condition.upper()}  (mass: {mass} kg)")
    print(f"{'='*55}")
    df_left  = pd.read_csv(f"data/raw/wearable/{participant}_{condition}_left_foot_data.csv",  header=0)
    df_right = pd.read_csv(f"data/raw/wearable/{participant}_{condition}_right_foot_data.csv", header=0)
    df_mocap = pd.read_csv(f"data/raw/mocap/{participant}_{condition}_mocap.csv",              header=[2, 5, 6])
    df_force = pd.read_csv(f"data/raw/forces/{participant}_{condition}_force.csv",
                           header=10, encoding='shift_jis')
    return dict(left=df_left, right=df_right, mocap=df_mocap, force=df_force,
                mass=mass, participant=participant, condition=condition)

# ============================================================
# 2. 列名整理
# ============================================================
def clean_mocap_columns(df):
    cols = []
    for col in df.columns:
        if col[2] == 'Frame':
            cols.append('Frame')
        elif 'Time' in col[2]:
            cols.append('Time (Seconds)')
        else:
            num = col[0].replace('Rigid Body', '').strip()
            cols.append(f"{num}_{col[1]}_{col[2]}")
    df.columns = cols
    return df

def clean_force_columns(df):
    return df.rename(columns={
        'Unnamed: 0': 'Time (Seconds)',
        '右-Fx': 'Right_Fx', '右-Fy': 'Right_Fy', '右-Fz': 'Right_Fz',
        '右-Mx': 'Right_Mx', '右-My': 'Right_My', '右-Mz': 'Right_Mz',
        '右-COPx': 'Right_COPx', '右-COPy': 'Right_COPy',
        '左-Fx': 'Left_Fx',  '左-Fy': 'Left_Fy',  '左-Fz': 'Left_Fz',
        '左-Mx': 'Left_Mx',  '左-My': 'Left_My',  '左-Mz': 'Left_Mz',
        '左-COPx': 'Left_COPx', '左-COPy': 'Left_COPy',
    })

# ============================================================
# 3. リサンプリング（一括補間に変更 → 3x faster）
# ============================================================
def process_resampling(df_input, sampling_interval=10):
    df = df_input.copy()
    exclude = ['Marker']
    target  = df.columns.difference(exclude)
    df[target] = df[target].interpolate(method='linear', axis=0)

    new_time = np.arange(0, df['ElapsedTime'].max(), sampling_interval)
    df_rs    = pd.DataFrame({'ElapsedTime': new_time})

    # Marker列はround-mergeで保持
    df_m = df[['ElapsedTime'] + exclude].dropna(subset=exclude, how='all').copy()
    df_m['_key'] = (df_m['ElapsedTime'] / sampling_interval).round().astype(int) * sampling_interval
    df_m = df_m.drop_duplicates('_key')
    df_rs['_key'] = df_rs['ElapsedTime'].round().astype(int)
    df_m['_key']  = df_m['_key'].round().astype(int)
    df_rs = df_rs.merge(df_m[['_key'] + exclude], on='_key', how='left').drop(columns='_key')

    # 連続値列を一括補間
    cont_cols = [c for c in df.columns if c not in set(['ElapsedTime'] + exclude)]
    if cont_cols:
        f = interp1d(df['ElapsedTime'], df[cont_cols].values,
                     axis=0, kind='linear', fill_value='extrapolate')
        df_rs[cont_cols] = f(new_time)

    df_rs['ElapsedTime'] /= 1000
    return df_rs.rename(columns={'ElapsedTime': 'Time (Seconds)'})

# ============================================================
# 4. ローパスフィルタ
# ============================================================
def apply_lowpass_filter(data, cutoff, fs, order=4):
    b, a = signal.butter(order, cutoff / (0.5 * fs), btype='low', analog=False)
    return signal.filtfilt(b, a, data, axis=0)

def process_smoothing_dataframe(df_input, fs=DEVICE_FREQ, cutoff=CUTOFF_FREQ, order=FILTER_ORDER):
    df   = df_input.copy()
    skip = ['Time (Seconds)', 'Marker']
    cols = [c for c in df.columns if c not in skip and 'Marker' not in c]
    df[cols] = df[cols].interpolate(method='linear', limit_direction='both').fillna(0)
    df[cols] = apply_lowpass_filter(df[cols].values, cutoff, fs, order)
    pcols = [c for c in cols if 'kPa' in c]
    if pcols:
        df[pcols] = df[pcols].clip(lower=0)
    return df

# ============================================================
# 5. 関節角度計算（完全ベクトル化 → 約120x faster）
# ============================================================
def _normalize_batch(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 1e-10, n, 1.0)

def _build_cs_long_axis_batch(e_long_raw, e_ml_ref):
    e_sup = _normalize_batch(e_long_raw)
    e_lat = _normalize_batch(e_ml_ref - np.einsum('ni,ni->n', e_ml_ref, e_sup)[:, None] * e_sup)
    e_ant = _normalize_batch(np.cross(e_sup, e_lat))
    e_sup = _normalize_batch(np.cross(e_lat, e_ant))
    return np.stack([e_lat, e_ant, e_sup], axis=-1)

def _build_cs_ant_axis_batch(e_ant_raw, e_sup_ref):
    e_ant = _normalize_batch(e_ant_raw)
    e_sup = _normalize_batch(e_sup_ref - np.einsum('ni,ni->n', e_sup_ref, e_ant)[:, None] * e_ant)
    e_lat = _normalize_batch(np.cross(e_ant, e_sup))
    e_sup = _normalize_batch(np.cross(e_lat, e_ant))
    return np.stack([e_lat, e_ant, e_sup], axis=-1)

def build_all_segment_cs_batch(pos_3d, side):
    """pos_3d: (N, 10, 3) → R_pelvis, R_thigh, R_shank, R_foot 各 (N, 3, 3)"""
    N = pos_3d.shape[0]
    if side == 'left':
        pos_3d = pos_3d.copy()
        pos_3d[:, :, 0] *= -1

    if side == 'right':
        p_asis = pos_3d[:, RI];  p_asis_c = pos_3d[:, LI]
        p_gt   = pos_3d[:, RGT]; p_knee   = pos_3d[:, RK]
        p_mall = pos_3d[:, RM];  p_toe    = pos_3d[:, RT]
    else:
        p_asis = pos_3d[:, LI];  p_asis_c = pos_3d[:, RI]
        p_gt   = pos_3d[:, LGT]; p_knee   = pos_3d[:, LK]
        p_mall = pos_3d[:, LM];  p_toe    = pos_3d[:, LT]

    e_ml     = _normalize_batch(p_asis - p_asis_c)
    R_pelvis = _build_cs_long_axis_batch(np.tile([0., 0., 1.], (N, 1)), e_ml)
    R_thigh  = _build_cs_long_axis_batch(p_gt - p_knee,  R_pelvis[:, :, 0])
    R_shank  = _build_cs_long_axis_batch(p_knee - p_mall, R_thigh[:, :, 0])
    R_foot   = _build_cs_ant_axis_batch(p_toe - p_mall,  R_shank[:, :, 2])
    return R_pelvis, R_thigh, R_shank, R_foot

def euler_xyz_batch(R_rel, joint='knee'):
    """(N, 3, 3) → (N, 3) XYZ オイラー角 [deg]"""
    flex = np.degrees(np.arctan2(-R_rel[:, 1, 2],  R_rel[:, 2, 2]))
    abd  = np.degrees(np.arcsin(np.clip(R_rel[:, 0, 2], -1., 1.)))
    rot  = np.degrees(np.arctan2(-R_rel[:, 0, 1],  R_rel[:, 0, 0]))
    return np.stack([flex, abd, rot], axis=-1) * SIGN_TABLE[joint]

def _mean_rotation_matrix(R_arr):
    R_mean = R_arr.mean(axis=0)
    U, _, Vt = LA.svd(R_mean)
    R_avg = U @ Vt
    if LA.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    return R_avg

def compute_static_joint_matrices(pos_frames, side):
    """(N, 10, 3) → (R_hip_ref, R_knee_ref, R_ankle_ref) 静的基準行列"""
    Rp, Rt, Rs, Rf = build_all_segment_cs_batch(pos_frames, side)
    return (
        _mean_rotation_matrix(np.matmul(Rp.transpose(0, 2, 1), Rt)),
        _mean_rotation_matrix(np.matmul(Rt.transpose(0, 2, 1), Rs)),
        _mean_rotation_matrix(np.matmul(Rs.transpose(0, 2, 1), Rf)),
    )

def calculate_angles_vectorized(df, target_cols, R_ref_right=None, R_ref_left=None):
    """全フレームの関節角度をベクトル化で計算（Pythonループなし）"""
    pos_3d = df[target_cols].values.reshape(-1, 10, 3)
    N = pos_3d.shape[0]
    if N == 0:
        return np.empty((0, JOINT_NO * 3))

    all_angles = np.zeros((N, JOINT_NO, 3))
    joint_keys = ['hip', 'knee', 'ankle']

    for s_idx, (side, R_refs) in enumerate([('right', R_ref_right), ('left', R_ref_left)]):
        Rp, Rt, Rs, Rf = build_all_segment_cs_batch(pos_3d, side)
        R_dyns = [
            np.matmul(Rp.transpose(0, 2, 1), Rt),
            np.matmul(Rt.transpose(0, 2, 1), Rs),
            np.matmul(Rs.transpose(0, 2, 1), Rf),
        ]
        for j_idx, (R_dyn, jkey) in enumerate(zip(R_dyns, joint_keys)):
            out_row = j_idx * 2 + s_idx
            R_rel = (np.matmul(R_refs[j_idx].T[None], R_dyn)
                     if R_refs is not None else R_dyn)
            all_angles[:, out_row] = euler_xyz_batch(R_rel, jkey)

    return all_angles.reshape(N, JOINT_NO * 3)

def process_mocap_data_target_calibration(df_target, df_ref):
    target_cols = []
    for i in range(1, 11):
        p = f"{i:03}"
        target_cols += [f"{p}_Position_X", f"{p}_Position_Y", f"{p}_Position_Z"]

    trigger_rows = df_ref[df_ref['Marker'] == 2]
    if trigger_rows.empty:
        raise ValueError("Marker == 2 が見つかりません")
    trig_t   = trigger_rows.iloc[0]['Time (Seconds)']
    s_t, e_t = trig_t - 20.0, trig_t - 15.0

    df_base = df_target[(df_target['Time (Seconds)'] >= s_t) &
                        (df_target['Time (Seconds)'] <= e_t)]
    if len(df_base) == 0:
        raise ValueError(f"静的区間データなし ({s_t:.1f}〜{e_t:.1f} sec)")

    pos_base    = df_base[target_cols].values.reshape(-1, 10, 3)
    R_ref_right = compute_static_joint_matrices(pos_base, 'right')
    R_ref_left  = compute_static_joint_matrices(pos_base, 'left')
    angles_all  = calculate_angles_vectorized(df_target, target_cols, R_ref_right, R_ref_left)

    joint_names = ["Right_Hip", "Left_Hip", "Right_Knee", "Left_Knee",
                   "Right_Ankle", "Left_Ankle"]
    dof_names   = ["Flex_Ext", "Abd_Add", "Int_Ext_Rot"]
    columns     = [f"{j}_{d}" for j in joint_names for d in dof_names]

    df_result = pd.DataFrame(angles_all, columns=columns)
    if 'Time (Seconds)' in df_target.columns:
        df_result.insert(0, 'Time (Seconds)', df_target['Time (Seconds)'].values)
    return df_result, columns

# ============================================================
# 6. 床反力 体重正規化 [%BW]
# ============================================================
def normalize_force_by_bodyweight(df_force, mass):
    cols  = ['Time (Seconds)', 'Right_Fx', 'Right_Fy', 'Right_Fz',
             'Left_Fx', 'Left_Fy', 'Left_Fz']
    fcols = [c for c in cols if c != 'Time (Seconds)']
    df    = df_force[cols].copy()
    df[fcols] /= mass * 9.81
    return df

# ============================================================
# 7. IMU オフセット除去（静止区間平均を引く）
# ============================================================
def remove_imu_offset(df_input):
    df = df_input.copy()
    df.columns = df.columns.str.replace('kPa', 'Pressure')
    imu_cols = [c for c in df.columns if 'Accel' in c or 'Gyro' in c]
    if not imu_cols:
        return df

    marker_cols = [c for c in df.columns if 'Marker' in c]
    start_t = 0.0
    if marker_cols:
        rows = df[(df[marker_cols] == 2).any(axis=1)]
        if not rows.empty:
            start_t = rows.iloc[0]['Time (Seconds)'] - 20.0

    mask   = (df['Time (Seconds)'] >= start_t) & (df['Time (Seconds)'] <= start_t + 5.0)
    static = df.loc[mask, imu_cols]
    if len(static) > 0:
        df[imu_cols] -= static.mean()
    else:
        print("  [WARNING] 静止区間が見つかりません（IMUオフセット除去スキップ）")
    return df

# ============================================================
# 8. データ同期・結合（一括補間に変更）
# ============================================================
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

    mk2_l  = df_left[df_left['Marker'] == 2]
    df_lf  = df_left.copy()
    df_rf  = df_right_r.copy()
    df_af  = df_angles.copy()

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
    t_comm   = np.arange(t_s, t_e, 1.0 / target_freq)
    df_merged = pd.DataFrame({'Time (Seconds)': t_comm})

    for df_src in [df_lf, df_rf, df_af, df_force_r]:
        tc   = 'Time (Seconds)'
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
        ts    = mk2_sync.iloc[0]['Time (Seconds)']
        df_out = df_merged[(df_merged['Time (Seconds)'] >= ts) &
                           (df_merged['Time (Seconds)'] <= ts + 300.0)].copy()
        df_out['Time (Seconds)'] -= ts
        return df_out.reset_index(drop=True)
    return df_merged

# ============================================================
# 9. ストライド検出・抽出
# ============================================================
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
    strides = [df.iloc[hs[i]:hs[i + 1]].copy()
               for i in range(1, len(hs) - 2)
               if min_dur <= (time[hs[i + 1]] - time[hs[i]]) <= max_dur]
    print(f"  [{side}] Accepted: {len(strides)} strides")
    return strides

# ============================================================
# 10. ストライド時間正規化（200点）
# ============================================================
def normalize_strides(stride_list, target_cols, n_points=200):
    x_new = np.linspace(0, 1, n_points)
    dfs, arrays = [], []
    for sdf in stride_list:
        x_old = np.linspace(0, 1, len(sdf))
        mat   = np.zeros((n_points, len(target_cols)))
        for j, col in enumerate(target_cols):
            if col in sdf.columns:
                f = interp1d(x_old, sdf[col].values, kind='linear', fill_value='extrapolate')
                mat[:, j] = f(x_new)
        new_df = pd.DataFrame(mat, columns=target_cols)
        new_df.insert(0, 'Gait Cycle (%)', np.linspace(0, 100, n_points))
        dfs.append(new_df)
        arrays.append(mat)
    ens = (np.array(arrays) if arrays else np.empty((0, n_points, len(target_cols))))
    return dfs, ens

# ============================================================
# 11. 外れ値除去（MAD法）
# ============================================================
def filter_outlier_strides_mad(ensemble, stride_dfs, n_mads=3.5, ratio_thresh=0.01):
    if len(ensemble) == 0:
        return ensemble, stride_dfs, np.array([], dtype=bool)
    median = np.median(ensemble, axis=0)
    sigma  = 1.4826 * np.median(np.abs(ensemble - median), axis=0)
    is_out = (ensemble > (median + n_mads * sigma)) | (ensemble < (median - n_mads * sigma))
    ratios = is_out.sum(axis=(1, 2)) / (ensemble.shape[1] * ensemble.shape[2])
    keep   = ratios <= ratio_thresh
    n_drop = (~keep).sum()
    if n_drop:
        print(f"  外れ値除去 (MAD): {n_drop}/{len(ensemble)} strides")
    return ensemble[keep], [d for i, d in enumerate(stride_dfs) if keep[i]], keep

# ============================================================
# 12. 左右統合
# ============================================================
def merge_left_right(left_ens, right_ens, left_cols, right_cols):
    left_d = left_ens.copy()
    for col, sign in [('Left_Accel_X', -1), ('Left_Gyro_Y', -1),
                      ('Left_Gyro_Z', -1), ('Left_Fx', -1)]:
        if col in left_cols:
            left_d[:, :, left_cols.index(col)] *= sign
    merged = np.concatenate([left_d, right_ens], axis=0)
    cols   = [c.replace('Left_', '').replace('Right_', '') for c in left_cols]
    print(f"  L:{left_ens.shape[0]} + R:{right_ens.shape[0]} = {merged.shape[0]} strides")
    return merged, cols

# ============================================================
# 13. 1被験者1条件の処理（並列ワーカー関数）
# ============================================================
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
        df_l_off     = remove_imu_offset(df_l)
        df_r_off     = remove_imu_offset(df_r)

        df_final = synchronize_merge_and_extract(
            df_l_off, df_r_off, df_angles, df_force_bw)

        l_strides = slice_strides(df_final, 'Left_Fz',  'Left')
        r_strides = slice_strides(df_final, 'Right_Fz', 'Right')

        l_dfs, l_ens = normalize_strides(l_strides, COLS_LEFT)
        r_dfs, r_ens = normalize_strides(r_strides, COLS_RIGHT)
        l_ens, l_dfs, _ = filter_outlier_strides_mad(l_ens, l_dfs)
        r_ens, r_dfs, _ = filter_outlier_strides_mad(r_ens, r_dfs)

        merged_ens, merged_cols = merge_left_right(l_ens, r_ens, COLS_LEFT, COLS_RIGHT)
        print(f"✓ {participant}_{condition}: {merged_ens.shape[0]} strides")
        return dict(participant=participant, condition=condition, mass=mass,
                    ensemble=merged_ens, columns=merged_cols)
    except Exception as e:
        print(f"✗ {participant}_{condition}: {e}")
        return None

# ============================================================
# 14. Phase 1: 全データのRAW処理（並列）
# ============================================================
def process_all_data_raw(output_dir='data/interim/raw_strides', n_workers=None):
    os.makedirs(output_dir, exist_ok=True)
    n_workers = n_workers or min(cpu_count(), 6)
    print(f"\n【Phase 1】並列処理開始 (workers={n_workers})")

    args = [(p, c, MASSES[i])
            for i, p in enumerate(PARTICIPANTS)
            for c in CONDITIONS]

    with Pool(n_workers) as pool:
        results = pool.map(_process_single, args)

    all_results = [r for r in results if r is not None]
    print(f"\n  成功: {len(all_results)}/{len(args)} 条件")

    for r in all_results:
        path = os.path.join(output_dir, f"{r['participant']}_{r['condition']}_raw.npz")
        np.savez(path, ensemble=r['ensemble'], columns=r['columns'],
                 participant=r['participant'], condition=r['condition'], mass=r['mass'])
    return all_results

# ============================================================
# 15. Phase 2: グローバル統計量の計算
#     train/val/test分割後は train のみ渡して再実行
# ============================================================
def _col_indices(cols):
    p_idx = [i for i, c in enumerate(cols) if 'Pressure' in c]
    i_idx = [i for i, c in enumerate(cols) if 'Accel' in c or 'Gyro' in c]
    a_idx = [i for i, c in enumerate(cols) if any(k in c for k in ('Hip', 'Knee', 'Ankle'))]
    return p_idx, i_idx, a_idx

def compute_global_stats(all_results, stats_path='data/processed/normalization_stats.npz'):
    print("\n【Phase 2】グローバル統計量を計算中...")
    cols = all_results[0]['columns']
    p_idx, i_idx, a_idx = _col_indices(cols)

    all_p, all_i = [], []
    for r in all_results:
        ens = r['ensemble']
        if ens.shape[0] == 0:
            continue
        all_p.append(ens[:, :, p_idx].reshape(-1, len(p_idx)))
        all_i.append(ens[:, :, i_idx].reshape(-1, len(i_idx)))

    all_p = np.concatenate(all_p, axis=0)
    all_i = np.concatenate(all_i, axis=0)
    p_min = all_p.min(axis=0)
    p_max = all_p.max(axis=0)
    i_std = np.where(all_i.std(axis=0) < 1e-8, 1.0, all_i.std(axis=0))

    np.savez(stats_path,
             pressure_min=p_min, pressure_max=p_max, imu_std=i_std,
             pressure_idx=np.array(p_idx), imu_idx=np.array(i_idx),
             angle_idx=np.array(a_idx), columns=np.array(cols))
    print(f"  保存: {stats_path}")
    print(f"  ★ train/val/test分割後は train のみで再実行してください")
    return dict(pressure_min=p_min, pressure_max=p_max, imu_std=i_std,
                pressure_idx=p_idx, imu_idx=i_idx, angle_idx=a_idx)

# ============================================================
# 16. Phase 3 & 4: 正規化適用 → 保存
# ============================================================
def apply_global_normalization(ensemble, stats):
    ens     = ensemble.copy()
    p_idx   = stats['pressure_idx']
    i_idx   = stats['imu_idx']
    a_idx   = stats['angle_idx']
    p_range = np.where((stats['pressure_max'] - stats['pressure_min']) < 1e-8,
                       1.0, stats['pressure_max'] - stats['pressure_min'])
    ens[:, :, p_idx] = (ens[:, :, p_idx] - stats['pressure_min']) / p_range
    ens[:, :, i_idx] = ens[:, :, i_idx] / stats['imu_std']
    ens[:, :, a_idx] = ens[:, :, a_idx] / ANGLE_SCALE
    return ens

def save_normalized_dataset(all_results, stats, output_dir='data/processed/normalized'):
    os.makedirs(output_dir, exist_ok=True)
    unique_p = sorted(set(r['participant'] for r in all_results))
    p_map    = {name: i for i, name in enumerate(unique_p)}
    c_map    = {'h': 0, 'm': 1, 'l': 2}

    all_ens, all_ids, all_conds = [], [], []
    for r in all_results:
        norm = apply_global_normalization(r['ensemble'], stats)
        np.savez(os.path.join(output_dir, f"{r['participant']}_{r['condition']}_norm.npz"),
                 ensemble=norm, columns=r['columns'],
                 participant=r['participant'], condition=r['condition'], mass=r['mass'])
        n = norm.shape[0]
        all_ens.append(norm)
        all_ids.append(np.full(n, p_map[r['participant']]))
        all_conds.append(np.full(n, c_map[r['condition']]))

    combined = np.concatenate(all_ens, axis=0)
    combined_path = os.path.join(output_dir, 'all_data_combined.npz')
    np.savez(combined_path,
             ensemble      = combined,
             subject_ids   = np.concatenate(all_ids,   axis=0),
             condition_ids = np.concatenate(all_conds, axis=0),
             columns       = all_results[0]['columns'],
             id_map        = p_map,
             condition_map = c_map,
             angle_scale   = ANGLE_SCALE)
    print(f"\n  保存: {combined_path}")
    print(f"  ensemble shape: {combined.shape}")
    print(f"  逆変換（関節角度）: pred_deg = pred_norm × {ANGLE_SCALE}")

# ============================================================
# メイン
# ============================================================
if __name__ == '__main__':
    import time
    t0 = time.time()

    all_results = process_all_data_raw(output_dir='data/interim/raw_strides', n_workers=None)
    if not all_results:
        raise RuntimeError("処理できたデータがありません")

    stats = compute_global_stats(all_results, stats_path='data/processed/normalization_stats.npz')

    print("\n【Phase 3 & 4】正規化の適用と保存")
    save_normalized_dataset(all_results, stats, output_dir='data/processed/normalized')

    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"  完了  (総処理時間: {elapsed/60:.1f} 分)")
    print(f"{'='*55}")