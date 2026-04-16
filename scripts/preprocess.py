"""
歩行データ前処理スクリプト（両脚対応版）

【データ構造】
  ensemble shape: (N_strides, 200, 52)
    [:, :,  0:26] = 同側足（ipsilateral）: Pressure×8, Accel×3, Gyro×3, Angles×9, GRF×3
    [:, :, 26:52] = 対側足（contralateral）: 同じ構造，列名に 'Contra_' プレフィックス

  片脚モデル  → ensemble[:, :, :26]  を使用
  両脚モデル  → ensemble[:, :, :]   をそのまま使用

【両脚ストライドの定義】
  右足HS→右足HS: ipsi=右足データ，contra=左足データ（同じ時間窓から抽出）
  左足HS→左足HS: ipsi=左足データ，contra=右足データ → 鏡映反転して右足基準に統一
  鏡映反転の対象軸:
    ipsi  側: Accel_X, Gyro_Y, Gyro_Z, Fx を ×-1
    contra側: Contra_Accel_X, Contra_Gyro_Y, Contra_Gyro_Z, Contra_Fx を ×-1
    ※鏡映後も「ipsi=同側，contra=対側」の関係は維持される

【正規化方針】
  IMU:
    ジャイロ  → 静止区間バイアス除去（条件ごと）+ グローバルstdスケーリング
    加速度    → グローバルstdスケーリングのみ（重力成分をモデルの姿勢推定基準として保持）
  足底圧力（Phase1）:
    体重[N]で割る → 被験者間体重差除去 + センサ間相対分布を保持
  足底圧力（Phase2）:
    全センサ・全データ・ipsi/contra合算の単一スカラーmaxで割る → [0, 1] にclip
  関節角度（出力ターゲット）: /180 固定定数スケーリング
  床反力: 体重正規化 [%BW] のまま
  外れ値除去: MAD法（52ch全チャンネルで判定）

【処理フロー】
  Phase 1: 全データの生ストライド抽出（52ch, 正規化前）
  Phase 2: グローバル統計量の計算（全データ or trainのみ）
  Phase 3: 正規化適用 → 保存

【モデル学習での使い方（例）】
  data = np.load('all_data_combined.npz', allow_pickle=True)
  # 片脚モデル（入力: ipsi 圧力+IMU, 出力: ipsi 角度+GRF）
  X    = data['ensemble'][:, :, :14]    # 圧力8 + 加速度3 + ジャイロ3
  y    = data['ensemble'][:, :, 14:26]  # 角度9 + GRF3
  # 両脚モデル（入力: ipsi + contra）
  X_bi = np.concatenate([data['ensemble'][:, :, :14],
                          data['ensemble'][:, :, 26:40]], axis=-1)  # 28ch
  # 所要時間を特徴ベクトルに concat
  dur  = data['durations']              # (N,) → CNN/LSTM 出力に concat してMLP
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

# ============================================================
# 列名定義
# ============================================================
# 生データ用（Left_/Right_ プレフィックスつき）
COLS_LEFT  = [
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

# 保存データ用（Side_プレフィックスなし）
_BASE_COLS     = [c.replace('Left_', '') for c in COLS_LEFT]  # 26ch
COLS_IPSI      = _BASE_COLS                                     # 同側  (index  0-25)
COLS_CONTRA    = ['Contra_' + c for c in _BASE_COLS]           # 対側  (index 26-51)
COLS_BILATERAL = COLS_IPSI + COLS_CONTRA                        # 両側  (52ch)

# 鏡映変換の対象軸と符号
_FLIP_COLS = {'Accel_X': -1, 'Gyro_Y': -1, 'Gyro_Z': -1, 'Fx': -1}

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
# 3. リサンプリング
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
# 5. 関節角度計算（NumPy完全ベクトル化）
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
    R_thigh  = _build_cs_long_axis_batch(p_gt - p_knee,   R_pelvis[:, :, 0])
    R_shank  = _build_cs_long_axis_batch(p_knee - p_mall,  R_thigh[:, :, 0])
    R_foot   = _build_cs_ant_axis_batch( p_toe - p_mall,   R_shank[:, :, 2])
    return R_pelvis, R_thigh, R_shank, R_foot

def euler_xyz_batch(R_rel, joint='knee'):
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
    Rp, Rt, Rs, Rf = build_all_segment_cs_batch(pos_frames, side)
    return (
        _mean_rotation_matrix(np.matmul(Rp.transpose(0, 2, 1), Rt)),
        _mean_rotation_matrix(np.matmul(Rt.transpose(0, 2, 1), Rs)),
        _mean_rotation_matrix(np.matmul(Rs.transpose(0, 2, 1), Rf)),
    )

def calculate_angles_vectorized(df, target_cols, R_ref_right=None, R_ref_left=None):
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
            R_rel   = (np.matmul(R_refs[j_idx].T[None], R_dyn)
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
# 7. IMU オフセット除去（ジャイロのみ）
# ============================================================
def remove_imu_offset(df_input):
    """
    ジャイロバイアス（ゼロドリフト）を静止区間の平均で除去する．
    加速度はオフセット除去しない（重力ベクトルをモデルの姿勢推定基準として保持）．
    """
    df = df_input.copy()
    df.columns = df.columns.str.replace('kPa', 'Pressure')
    gyro_cols = [c for c in df.columns if 'Gyro' in c]
    if not gyro_cols:
        return df
    marker_cols = [c for c in df.columns if 'Marker' in c]
    start_t = 0.0
    if marker_cols:
        rows = df[(df[marker_cols] == 2).any(axis=1)]
        if not rows.empty:
            start_t = rows.iloc[0]['Time (Seconds)'] - 20.0
    mask   = (df['Time (Seconds)'] >= start_t) & (df['Time (Seconds)'] <= start_t + 5.0)
    static = df.loc[mask, gyro_cols]
    if len(static) > 0:
        df[gyro_cols] -= static.mean()
        print(f"  ジャイロバイアス除去 ({len(static)} フレーム基準，加速度は重力基準保持)")
    else:
        print("  [WARNING] 静止区間なし（ジャイロバイアス除去スキップ）")
    return df

# ============================================================
# 8. データ同期・結合
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
    """
    df_final（両足列 Left_* + Right_* を含む）から基準足のHSでストライドを切り出す．
    切り出した各 DataFrame には同側・対側両方のデータが含まれる．

    Returns
    -------
    strides:   DataFrame リスト（両足列を含む）
    durations: ndarray (N,)  各ストライドの所要時間 [sec]
    """
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

# ============================================================
# 10. ストライド時間正規化（両脚対応: 52ch）
# ============================================================
def normalize_strides_bilateral(stride_list, ipsi_raw_cols, contra_raw_cols, n_points=200):
    """
    同側・対側両方のデータを200点に時間正規化する．

    Parameters
    ----------
    stride_list:     スライス済み DataFrame リスト（両足列を含む df_final のスライス）
    ipsi_raw_cols:   同側足の生列名  (例: COLS_RIGHT for right strides)
    contra_raw_cols: 対側足の生列名  (例: COLS_LEFT  for right strides)

    Returns
    -------
    dfs: DataFrame リスト (columns = COLS_BILATERAL)
    ens: ndarray   (N, 200, 52)
           [:, :,  0:26] = ipsi  (COLS_IPSI)
           [:, :, 26:52] = contra (COLS_CONTRA)
    """
    x_new        = np.linspace(0, 1, n_points)
    all_raw_cols = ipsi_raw_cols + contra_raw_cols  # 52 列の生データ列名
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

# ============================================================
# 11. 外れ値除去（MAD法）
# ============================================================
def filter_outlier_strides_mad(ensemble, stride_dfs, durations=None,
                                n_mads=3.5, ratio_thresh=0.01):
    """
    52ch 全チャンネルで MAD 判定する．
    同側・対側どちらかに異常があるストライドを除去できる．

    Parameters
    ----------
    durations: ndarray or None  keep_mask を同期して適用する
    """
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

# ============================================================
# 12. 両脚統合（鏡映反転 + 結合）
# ============================================================
def merge_bilateral(left_ens, right_ens, left_dur=None, right_dur=None):
    """
    右足ストライドと左足ストライドを統合する．
    左足ストライドは鏡映変換（X軸反転）で右足基準に統一する．

    【鏡映変換の対象軸】
    _FLIP_COLS = {Accel_X, Gyro_Y, Gyro_Z, Fx} を ×-1
    ipsi 側と contra 側の両方に同じ変換を適用する．

    【contra 側にも同じ変換を適用する理由】
    鏡映後の座標系では，対側足の外側方向（lateral axis）も
    ipsi 足と同様に符号が反転するため，同一の軸反転が必要．
    これにより「ipsi=同側，contra=対側」の幾何的関係が左右間で一致する．

    Returns
    -------
    merged:       (N_R + N_L, 200, 52)
    COLS_BILATERAL: 52ch 列名リスト
    merged_dur:   ndarray or None
    """
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

# ============================================================
# 13. 1被験者1条件の処理（並列ワーカー）
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

        # ジャイロバイアス除去（加速度は重力成分保持のため対象外）
        df_l_off = remove_imu_offset(df_l)
        df_r_off = remove_imu_offset(df_r)

        # 足底圧力 体重正規化（Phase1）
        # 全8センサを同じ体重[N]で割ることで，被験者間体重差を除去しつつ
        # 踵・土踏まず・つま先の空間的圧力分布（センサ間の相対バランス）を保持する
        bw = mass * 9.81
        for df_dev in [df_l_off, df_r_off]:
            pcols = [c for c in df_dev.columns if 'Pressure' in c]
            if pcols:
                df_dev[pcols] /= bw

        # 同期・結合・300秒抽出
        # df_final には Left_* / Right_* 両方の列が含まれる
        df_final = synchronize_merge_and_extract(
            df_l_off, df_r_off, df_angles, df_force_bw)

        # ストライド切り出し（両足データを含む df_final のスライス）
        r_strides, r_dur = slice_strides(df_final, 'Right_Fz', 'Right')
        l_strides, l_dur = slice_strides(df_final, 'Left_Fz',  'Left')

        # 時間正規化（200点，52ch）
        # 右ストライド: ipsi=Right, contra=Left
        r_dfs, r_ens = normalize_strides_bilateral(r_strides, COLS_RIGHT, COLS_LEFT)
        # 左ストライド: ipsi=Left,  contra=Right
        l_dfs, l_ens = normalize_strides_bilateral(l_strides, COLS_LEFT,  COLS_RIGHT)

        # 外れ値除去（52ch 全チャンネルで MAD 判定）
        r_ens, r_dfs, _, r_dur = filter_outlier_strides_mad(r_ens, r_dfs, r_dur)
        l_ens, l_dfs, _, l_dur = filter_outlier_strides_mad(l_ens, l_dfs, l_dur)

        # 鏡映反転 + 左右統合 → (N_total, 200, 52)
        merged_ens, merged_cols, merged_dur = merge_bilateral(l_ens, r_ens, l_dur, r_dur)

        print(f"✓ {participant}_{condition}: {merged_ens.shape[0]} strides (52ch bilateral)")
        return dict(participant=participant, condition=condition, mass=mass,
                    ensemble=merged_ens, columns=merged_cols, durations=merged_dur)
    except Exception as e:
        import traceback
        print(f"✗ {participant}_{condition}: {e}")
        traceback.print_exc()
        return None

# ============================================================
# 14. Phase 1: 全データのRAW処理（並列）
# ============================================================
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

# ============================================================
# 15. Phase 2: グローバル統計量の計算
#     train/val/test 分割後は train のみ渡して再実行
# ============================================================
def _col_indices(cols):
    """
    COLS_BILATERAL (52ch) のモダリティ別インデックスを返す．
    ipsi / contra に同一の統計量を適用するため両方を返す．
    """
    p_idx  = [i for i, c in enumerate(cols) if 'Pressure' in c and 'Contra' not in c]
    i_idx  = [i for i, c in enumerate(cols) if ('Accel' in c or 'Gyro' in c) and 'Contra' not in c]
    a_idx  = [i for i, c in enumerate(cols) if any(k in c for k in ('Hip','Knee','Ankle')) and 'Contra' not in c]
    cp_idx = [i for i, c in enumerate(cols) if 'Pressure' in c and 'Contra' in c]
    ci_idx = [i for i, c in enumerate(cols) if ('Accel' in c or 'Gyro' in c) and 'Contra' in c]
    ca_idx = [i for i, c in enumerate(cols) if any(k in c for k in ('Hip','Knee','Ankle')) and 'Contra' in c]
    return p_idx, i_idx, a_idx, cp_idx, ci_idx, ca_idx

def compute_global_stats(all_results, stats_path='data/processed/normalization_stats.npz'):
    """
    グローバル正規化統計量を計算して保存する．

    【足底圧力】
    ipsi + contra を合わせた全センサ・全データの単一スカラーmax を使用する．
    センサごとの独立 min-max は踵/土踏まず/つま先の空間的分布を破壊するため使わない．

    【IMU】
    ipsi 側の軸ごと std を計算し，contra にも同じ値を適用する（同一物理量のため）．
    """
    print("\n【Phase 2】グローバル統計量を計算中...")
    cols = all_results[0]['columns']
    p_idx, i_idx, a_idx, cp_idx, ci_idx, ca_idx = _col_indices(cols)

    all_p, all_i = [], []
    for r in all_results:
        ens = r['ensemble']   # (N, 200, 52)
        if ens.shape[0] == 0:
            continue
        # ipsi + contra を合わせてスカラーmax を求める（同一物理量のため）
        p_combined = ens[:, :, p_idx + cp_idx].reshape(-1, len(p_idx) + len(cp_idx))
        i_ipsi     = ens[:, :, i_idx         ].reshape(-1, len(i_idx))
        all_p.append(p_combined)
        all_i.append(i_ipsi)

    all_p = np.concatenate(all_p, axis=0)
    all_i = np.concatenate(all_i, axis=0)

    p_global_max = float(all_p.max())
    p_global_min = 0.0  # 体重正規化後，物理的に 0 以上が保証される

    i_std = all_i.std(axis=0)
    i_std = np.where(i_std < 1e-8, 1.0, i_std)

    np.savez(stats_path,
             pressure_global_max = np.array(p_global_max),
             pressure_global_min = np.array(p_global_min),
             imu_std             = i_std,
             p_idx  = np.array(p_idx),  i_idx  = np.array(i_idx),  a_idx  = np.array(a_idx),
             cp_idx = np.array(cp_idx), ci_idx = np.array(ci_idx), ca_idx = np.array(ca_idx),
             columns = np.array(cols))

    print(f"  圧力 global_max (ipsi+contra 合算): {p_global_max:.6f}")
    print(f"  IMU std 6軸 (ipsi 基準):            {i_std.round(4)}")
    print(f"  保存: {stats_path}")
    print(f"  ★ train/val/test 分割後は train のみで再実行してください")

    return dict(pressure_global_max=p_global_max, pressure_global_min=p_global_min,
                imu_std=i_std,
                p_idx=p_idx,   i_idx=i_idx,   a_idx=a_idx,
                cp_idx=cp_idx, ci_idx=ci_idx, ca_idx=ca_idx)

# ============================================================
# 16. Phase 3 & 4: 正規化適用 → 保存
# ============================================================
def apply_global_normalization(ensemble, stats):
    """
    52ch ensemble にグローバル統計量を適用する．

    足底圧力（ipsi+contra）: 単一スカラーmaxで割り [0,1] にclip
    IMU（ipsi+contra）:      ipsi の std を両方に適用
    関節角度（ipsi+contra）: /180 固定定数
    床反力:                  体重正規化済み [%BW] のままスケーリングしない
    """
    ens   = ensemble.copy()
    p_max = float(stats['pressure_global_max'])
    p_min = float(stats['pressure_global_min'])
    p_range = max(p_max - p_min, 1e-8)
    i_std   = stats['imu_std']  # (6,)

    # 足底圧力（ipsi + contra）
    for idx in [stats['p_idx'], stats['cp_idx']]:
        ens[:, :, idx] = np.clip((ens[:, :, idx] - p_min) / p_range, 0.0, 1.0)

    # IMU（ipsi と contra に同じ std を適用）
    ens[:, :, stats['i_idx']]  /= i_std
    ens[:, :, stats['ci_idx']] /= i_std

    # 関節角度（ipsi + contra）
    for idx in [stats['a_idx'], stats['ca_idx']]:
        ens[:, :, idx] /= ANGLE_SCALE

    return ens

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
    print(f"\n  【使い方】")
    print(f"  片脚モデル : X = ensemble[:, :,  0:14]  (ipsi  圧力8 + IMU6)")
    print(f"  両脚モデル : X_ipsi   = ensemble[:, :,  0:14]  (ipsi  圧力8 + IMU6)")
    print(f"               X_contra = ensemble[:, :, 26:40]  (contra 圧力8 + IMU6)")
    print(f"  ターゲット : y = ensemble[:, :, 14:26]  (ipsi  角度9 + GRF3)")
    print(f"  所要時間   : dur = durations  → 時系列特徴ベクトルに concat してMLP")
    print(f"  逆変換     : pred_deg = pred_norm × {ANGLE_SCALE}")

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