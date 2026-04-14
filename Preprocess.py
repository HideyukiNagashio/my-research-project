"""
歩行データ前処理スクリプト（深層学習用）

【正規化方針（ipynbからの主な変更点）】
1. IMU:
   - オフセット除去: 静止区間(Marker==2前20〜15秒)の平均を引く（条件ごとでOK）
   - スケーリング: グローバルstdで割る（全データから計算，train分割後はtrainのみで再計算）
   - ※ 旧実装は条件ごとにmean・stdの両方を計算していたため，条件間の振幅差が消えていた

2. 足底圧力:
   - ネガティブクリップのみ実施（正規化はPhase2でグローバルmin-maxを適用）
   - ※ 旧実装は条件ごとmin-maxのため，歩行速度による荷重差が消えていた

3. 関節角度（出力ターゲット）:
   - 固定定数 /180 でスケーリング（逆変換 ×180 のみでOK）
   - ※ Z-scoreは非正規分布な関節角度には不適切

4. 外れ値除去:
   - 粒度は条件ごと（被験者×条件）のまま維持（正しい）
   - stdの代わりにMADを使用（外れ値がstd自体を膨らませるのを防ぐ）
   - ※ 全体基準にすると歩行速度差・被験者差が分散に混入して判定が甘くなる

【処理フロー】
Phase 1: 全データの生ストライド抽出（正規化前）
Phase 2: グローバル正規化統計量の計算（全データ or trainのみ）
Phase 3: 正規化の適用
Phase 4: 保存

【train/val/test分割について】
現時点では分割未実施のため，全データでstatsを計算する．
LOSO等で分割後は compute_global_stats() に train被験者のデータのみ渡して
stats_all.npz を再計算・上書きすること．
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal
from numpy import linalg as LA

# ============================================================
# 実験設定
# ============================================================
PARTICIPANTS = ['oba', 'ono', 'pon', 'kuno', 'john', 'konan',
                'obara', 'fukuzawa', 'kiuchi', 'yanaze', 'adachi', 'iwasaki']
MASSES = [60.9, 63.8, 68.7, 65.9, 77, 74, 63.8, 64.5, 73.2, 53.5, 70.7, 47.9]
CONDITIONS = ['h', 'm', 'l']

DEVICE_FREQ = 100
MOCAP_FREQ  = 250
FORCE_FREQ  = 1000

CUTOFF_FREQ  = 6
FILTER_ORDER = 4

ANGLE_SCALE = 180.0  # 関節角度の固定スケーリング定数（/180 → −1〜1付近）

# ============================================================
# 1. データ読み込み
# ============================================================
def load_participant_data(participant, condition, mass):
    print(f"\n{'='*60}")
    print(f"Processing: {participant} - Condition: {condition.upper()} (Mass: {mass} kg)")
    print(f"{'='*60}")

    file_path_left  = f"WearableDevices/{participant}_{condition}_left_foot_data.csv"
    file_path_right = f"WearableDevices/{participant}_{condition}_right_foot_data.csv"
    file_path_mocap = f"MotionCaptures/{participant}_{condition}_mocap.csv"
    file_path_force = f"3DGroundForces/{participant}_{condition}_force.csv"

    df_left  = pd.read_csv(file_path_left,  header=0)
    df_right = pd.read_csv(file_path_right, header=0)
    df_mocap = pd.read_csv(file_path_mocap, header=[2, 5, 6])
    df_force = pd.read_csv(file_path_force, header=10, encoding='shift_jis')

    return {'left': df_left, 'right': df_right,
            'mocap': df_mocap, 'force': df_force,
            'mass': mass, 'participant': participant, 'condition': condition}

# ============================================================
# 2. 列名整理
# ============================================================
def clean_mocap_columns(df_mocap):
    new_columns = []
    for col in df_mocap.columns:
        if col[2] == 'Frame':
            new_columns.append('Frame')
        elif 'Time' in col[2]:
            new_columns.append('Time (Seconds)')
        else:
            body_num = col[0].replace('Rigid Body', '').strip()
            new_columns.append(f"{body_num}_{col[1]}_{col[2]}")
    df_mocap.columns = new_columns
    return df_mocap

def clean_force_columns(df_force):
    columns_mapping = {
        'Unnamed: 0': 'Time (Seconds)',
        '右-Fx': 'Right_Fx', '右-Fy': 'Right_Fy', '右-Fz': 'Right_Fz',
        '右-Mx': 'Right_Mx', '右-My': 'Right_My', '右-Mz': 'Right_Mz',
        '右-COPx': 'Right_COPx', '右-COPy': 'Right_COPy',
        '左-Fx': 'Left_Fx', '左-Fy': 'Left_Fy', '左-Fz': 'Left_Fz',
        '左-Mx': 'Left_Mx', '左-My': 'Left_My', '左-Mz': 'Left_Mz',
        '左-COPx': 'Left_COPx', '左-COPy': 'Left_COPy',
    }
    return df_force.rename(columns=columns_mapping)

# ============================================================
# 3. リサンプリング
# ============================================================
def process_resampling(df_input, sampling_interval=10):
    df = df_input.copy()

    exclude_cols = ['Marker']
    target_cols  = df.columns.difference(exclude_cols)
    df[target_cols] = df[target_cols].interpolate(method='linear', axis=0)

    time_min = 0
    time_max = df['ElapsedTime'].max()
    new_time = np.arange(time_min, time_max, sampling_interval)
    df_resampled = pd.DataFrame({'ElapsedTime': new_time})

    if exclude_cols:
        df_markers = df[['ElapsedTime'] + exclude_cols].dropna(subset=exclude_cols, how='all').copy()
        df_markers['ElapsedTime_rounded'] = (df_markers['ElapsedTime'] / sampling_interval).round() * sampling_interval
        df_markers = df_markers.drop_duplicates(subset=['ElapsedTime_rounded'])

        df_resampled['MergeKey'] = df_resampled['ElapsedTime'].round().astype(int)
        df_markers['MergeKey']   = df_markers['ElapsedTime_rounded'].round().astype(int)
        df_resampled = pd.merge(df_resampled, df_markers[['MergeKey'] + exclude_cols],
                                on='MergeKey', how='left')
        df_resampled = df_resampled.drop(columns=['MergeKey'])

    columns_to_exclude = ['ElapsedTime'] + exclude_cols
    for column in df.columns:
        if column in columns_to_exclude:
            continue
        f = interp1d(df['ElapsedTime'], df[column], kind='linear', fill_value='extrapolate')
        df_resampled[column] = f(new_time)

    df_resampled['ElapsedTime'] = df_resampled['ElapsedTime'] / 1000
    df_resampled = df_resampled.rename(columns={'ElapsedTime': 'Time (Seconds)'})
    return df_resampled

# ============================================================
# 4. ローパスフィルタ
# ============================================================
def apply_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, data, axis=0)

def process_smoothing_dataframe(df_input, fs=DEVICE_FREQ, cutoff=CUTOFF_FREQ, order=FILTER_ORDER):
    df_smooth = df_input.copy()
    exclude_keywords = ['Time (Seconds)', 'Marker']
    filter_target_cols = [col for col in df_smooth.columns
                          if col not in exclude_keywords and 'Marker' not in col]

    df_smooth[filter_target_cols] = df_smooth[filter_target_cols].interpolate(
        method='linear', limit_direction='both')
    df_smooth[filter_target_cols] = df_smooth[filter_target_cols].fillna(0)
    df_smooth[filter_target_cols] = apply_lowpass_filter(
        df_smooth[filter_target_cols].values, cutoff=cutoff, fs=fs, order=order)

    # 圧力の負値クリップ（フィルタ後のアーティファクト除去）
    pressure_cols = [col for col in filter_target_cols if 'kPa' in col]
    if pressure_cols:
        df_smooth[pressure_cols] = df_smooth[pressure_cols].mask(
            df_smooth[pressure_cols] < 0, 0)
    return df_smooth

# ============================================================
# 5. 関節角度計算（ipynbと同一）
# ============================================================
RI  = 0; RGT = 1; RK = 2; RM = 3; RT = 4
LI  = 5; LGT = 6; LK = 7; LM = 8; LT = 9
JOINT_NO = 6

def _normalize(v):
    n = LA.norm(v)
    return v / n if n > 1e-10 else v.copy()

def _mean_rotation_matrix(R_list):
    R_mean = np.mean(np.array(R_list), axis=0)
    U, _, Vt = LA.svd(R_mean)
    R_avg = U @ Vt
    if LA.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    return R_avg

def _build_cs_long_axis(e_long_raw, e_ml_ref):
    e_sup = _normalize(e_long_raw)
    e_lat = e_ml_ref - np.dot(e_ml_ref, e_sup) * e_sup
    e_lat = _normalize(e_lat)
    e_ant = _normalize(np.cross(e_sup, e_lat))
    e_sup = _normalize(np.cross(e_lat, e_ant))
    return np.column_stack([e_lat, e_ant, e_sup])

def _build_cs_ant_axis(e_ant_raw, e_sup_ref):
    e_ant = _normalize(e_ant_raw)
    e_sup = e_sup_ref - np.dot(e_sup_ref, e_ant) * e_ant
    e_sup = _normalize(e_sup)
    e_lat = _normalize(np.cross(e_ant, e_sup))
    e_sup = _normalize(np.cross(e_lat, e_ant))
    return np.column_stack([e_lat, e_ant, e_sup])

def build_all_segment_cs(pos, side):
    if side == 'left':
        pos = pos.copy()
        pos[:, 0] *= -1
    if side == 'right':
        idx = {'asis': RI, 'gt': RGT, 'knee': RK, 'mall': RM, 'toe': RT, 'asis_c': LI}
    else:
        idx = {'asis': LI, 'gt': LGT, 'knee': LK, 'mall': LM, 'toe': LT, 'asis_c': RI}

    p_asis   = pos[idx['asis']];  p_asis_c = pos[idx['asis_c']]
    p_gt     = pos[idx['gt']];    p_knee   = pos[idx['knee']]
    p_mall   = pos[idx['mall']];  p_toe    = pos[idx['toe']]

    e_ml_pelvis = _normalize(p_asis - p_asis_c)
    R_pelvis = _build_cs_long_axis(np.array([0., 0., 1.]), e_ml_pelvis)
    R_thigh  = _build_cs_long_axis(p_gt   - p_knee, R_pelvis[:, 0])
    R_shank  = _build_cs_long_axis(p_knee - p_mall, R_thigh[:, 0])
    R_foot   = _build_cs_ant_axis(p_toe   - p_mall, R_shank[:, 2])
    return R_pelvis, R_thigh, R_shank, R_foot

SIGN_TABLE = {
    'hip':   np.array([ 1., 1., 1.]),
    'knee':  np.array([-1., 1., 1.]),
    'ankle': np.array([ 1., 1., 1.]),
}

def rotation_to_euler_xyz(R_rel, joint='knee'):
    flex = np.degrees(np.arctan2(-R_rel[1, 2],  R_rel[2, 2]))
    abd  = np.degrees(np.arcsin(np.clip(R_rel[0, 2], -1., 1.)))
    rot  = np.degrees(np.arctan2(-R_rel[0, 1],  R_rel[0, 0]))
    return np.array([flex, abd, rot]) * SIGN_TABLE[joint]

class CalculateAngle:
    def angles_one_side(self, pos, side):
        Rp, Rt, Rs, Rf = build_all_segment_cs(pos, side)
        return np.stack([
            rotation_to_euler_xyz(Rp.T @ Rt, 'hip'),
            rotation_to_euler_xyz(Rt.T @ Rs, 'knee'),
            rotation_to_euler_xyz(Rs.T @ Rf, 'ankle'),
        ], axis=0)

    def angles(self, pos):
        right = self.angles_one_side(pos, 'right')
        left  = self.angles_one_side(pos, 'left')
        return np.array([right[0], left[0], right[1], left[1], right[2], left[2]])

def compute_static_joint_matrices(pos_frames, side):
    R_hips, R_knees, R_ankles = [], [], []
    for pos in pos_frames:
        Rp, Rt, Rs, Rf = build_all_segment_cs(pos, side)
        R_hips.append(Rp.T @ Rt)
        R_knees.append(Rt.T @ Rs)
        R_ankles.append(Rs.T @ Rf)
    return (_mean_rotation_matrix(R_hips),
            _mean_rotation_matrix(R_knees),
            _mean_rotation_matrix(R_ankles))

def calculate_angles_from_positions(df, target_cols, R_ref_right=None, R_ref_left=None):
    pos_flat = df[target_cols].values
    n_frames = pos_flat.shape[0]
    if n_frames == 0:
        return np.empty((0, 18))
    pos_3d = pos_flat.reshape(n_frames, 10, 3)
    calc   = CalculateAngle()
    all_angles = np.zeros((n_frames, JOINT_NO, 3))
    use_calib  = (R_ref_right is not None) and (R_ref_left is not None)
    joint_keys = ['hip', 'knee', 'ankle']

    for f, pos in enumerate(pos_3d):
        if not use_calib:
            all_angles[f] = calc.angles(pos)
        else:
            for s_idx, (side, R_refs) in enumerate(
                    [('right', R_ref_right), ('left', R_ref_left)]):
                Rp, Rt, Rs, Rf = build_all_segment_cs(pos, side)
                R_dyns = [Rp.T @ Rt, Rt.T @ Rs, Rs.T @ Rf]
                for j_idx, (R_dyn, R_ref, jkey) in enumerate(
                        zip(R_dyns, R_refs, joint_keys)):
                    out_row = j_idx * 2 + s_idx
                    all_angles[f, out_row] = rotation_to_euler_xyz(R_ref.T @ R_dyn, jkey)
    return all_angles.reshape(n_frames, JOINT_NO * 3)

def process_mocap_data_target_calibration(df_target, df_ref):
    target_cols = []
    for i in range(1, 11):
        prefix = f"{i:03}"
        target_cols.extend([f"{prefix}_Position_X", f"{prefix}_Position_Y", f"{prefix}_Position_Z"])

    trigger_rows = df_ref[df_ref['Marker'] == 2]
    if trigger_rows.empty:
        raise ValueError("df_ref に Marker == 2 のデータが見つかりません．")
    trigger_time = trigger_rows.iloc[0]['Time (Seconds)']
    start_time   = trigger_time - 20.0
    end_time     = trigger_time - 15.0
    print(f"トリガー検知: {trigger_time:.4f} sec  静的基準区間: {start_time:.2f}〜{end_time:.2f} sec")

    mask    = (df_target['Time (Seconds)'] >= start_time) & (df_target['Time (Seconds)'] <= end_time)
    df_base = df_target.loc[mask].copy()
    if len(df_base) == 0:
        raise ValueError("静的区間のデータが存在しません．")

    pos_base    = df_base[target_cols].values.reshape(-1, 10, 3)
    R_ref_right = compute_static_joint_matrices(pos_base, 'right')
    R_ref_left  = compute_static_joint_matrices(pos_base, 'left')

    angles_all  = calculate_angles_from_positions(df_target, target_cols, R_ref_right, R_ref_left)

    joint_names = ["Right_Hip", "Left_Hip", "Right_Knee", "Left_Knee", "Right_Ankle", "Left_Ankle"]
    dof_names   = ["Flex_Ext", "Abd_Add", "Int_Ext_Rot"]
    columns     = [f"{j}_{d}" for j in joint_names for d in dof_names]

    df_result = pd.DataFrame(angles_all, columns=columns)
    if 'Time (Seconds)' in df_target.columns:
        df_result.insert(0, 'Time (Seconds)', df_target['Time (Seconds)'].values)

    return df_result, {'right': R_ref_right, 'left': R_ref_left}, columns

# ============================================================
# 6. 床反力の体重正規化 [%BW]  ← 変更なし
# ============================================================
def normalize_force_by_bodyweight(df_force, mass):
    columns       = ['Time (Seconds)', 'Right_Fx', 'Right_Fy', 'Right_Fz',
                     'Left_Fx', 'Left_Fy', 'Left_Fz']
    force_columns = ['Right_Fx', 'Right_Fy', 'Right_Fz',
                     'Left_Fx',  'Left_Fy',  'Left_Fz']
    body_weight        = mass * 9.81
    df_normalized      = df_force[columns].copy()
    df_normalized[force_columns] /= body_weight
    return df_normalized

# ============================================================
# 7. IMU オフセット除去（条件ごとでOK）
#    静止区間（Marker==2 前20〜15秒）の平均をゼロ基準にする
#    ※ スケーリング（stdで割る）はPhase2でグローバルに実施
# ============================================================
def remove_imu_offset(df_input):
    """
    静止立位区間の平均をIMUデータから引く（オフセット除去）．
    目的: センサー装着誤差・ドリフトによる個体間オフセットの除去．
    スケーリング（stdで割る）はグローバル統計量で別途実施するため，ここでは不要．
    """
    df = df_input.copy()
    df.columns = df.columns.str.replace('kPa', 'Pressure')

    imu_cols = [col for col in df.columns
                if 'Accel' in col or 'Gyro' in col]

    marker_cols = [col for col in df.columns if 'Marker' in col]
    start_time  = 0.0

    if marker_cols:
        marker_2_rows = df[(df[marker_cols] == 2).any(axis=1)]
        if not marker_2_rows.empty:
            trigger_t  = marker_2_rows.iloc[0]['Time (Seconds)']
            start_time = trigger_t - 20.0  # キャリブレーション開始
            end_time   = trigger_t - 15.0  # キャリブレーション終了
        else:
            end_time = start_time + 5.0
    else:
        end_time = start_time + 5.0

    mask        = (df['Time (Seconds)'] >= start_time) & (df['Time (Seconds)'] <= end_time)
    df_static   = df.loc[mask]

    if len(df_static) == 0:
        print("  [WARNING] 静止区間が見つかりません．オフセット除去をスキップします．")
        return df

    offsets = df_static[imu_cols].mean()
    df[imu_cols] = df[imu_cols] - offsets

    print(f"  IMUオフセット除去完了 ({len(df_static)} フレームを基準に使用)")
    return df

# ============================================================
# 8. データ同期と結合（変更なし）
# ============================================================
def calculate_fine_offset_pressure(df_target, df_ref, col_target_pressure_list,
                                   col_ref_name, t_start, duration=300, fs=100):
    t_end = t_start + duration
    mask_tgt = (df_target['Time (Seconds)'] >= t_start) & (df_target['Time (Seconds)'] <= t_end)
    df_t = df_target.loc[mask_tgt].copy()
    mask_ref = (df_ref['Time (Seconds)'] >= t_start) & (df_ref['Time (Seconds)'] <= t_end)
    df_r = df_ref.loc[mask_ref].copy()

    if len(df_t) < fs or len(df_r) < fs:
        return 0.0

    t_min = max(df_t['Time (Seconds)'].min(), df_r['Time (Seconds)'].min())
    t_max = min(df_t['Time (Seconds)'].max(), df_r['Time (Seconds)'].max())
    if t_min >= t_max:
        return 0.0

    common_t = np.arange(t_min, t_max, 1.0 / fs)
    pressure_sum = df_t[col_target_pressure_list].sum(axis=1)
    f_tgt = interp1d(df_t['Time (Seconds)'], pressure_sum,
                     kind='linear', fill_value=0, bounds_error=False)
    sig_tgt = f_tgt(common_t)

    f_ref = interp1d(df_r['Time (Seconds)'], df_r[col_ref_name],
                     kind='linear', fill_value=0, bounds_error=False)
    sig_ref = f_ref(common_t)

    sig_tgt_norm = (sig_tgt - np.mean(sig_tgt)) / (np.std(sig_tgt) + 1e-6)
    sig_ref_norm = (sig_ref - np.mean(sig_ref)) / (np.std(sig_ref) + 1e-6)

    correlation = signal.correlate(sig_ref_norm, sig_tgt_norm, mode='full')
    lags        = signal.correlation_lags(len(sig_ref_norm), len(sig_tgt_norm), mode='full')
    return lags[np.argmax(correlation)] / fs

def synchronize_merge_and_extract(df_left, df_right, df_angles, df_force, target_freq=100):
    trigger_marker = 1
    marker_rows_l = df_left[df_left['Marker'] == trigger_marker]
    t_marker_left_1 = marker_rows_l.iloc[0]['Time (Seconds)'] if not marker_rows_l.empty else 0.0

    df_right_rough = df_right.copy()
    marker_rows_r  = df_right[df_right['Marker'] == trigger_marker]
    if not marker_rows_r.empty:
        offset_r_rough = t_marker_left_1 - marker_rows_r.iloc[0]['Time (Seconds)']
        df_right_rough['Time (Seconds)'] += offset_r_rough

    df_force_rough = df_force.copy()
    df_force_rough['Time (Seconds)'] += t_marker_left_1

    cols_Pressure_left  = [f'Left_Pressure_{i}'  for i in range(1, 9)]
    cols_Pressure_right = [f'Right_Pressure_{i}' for i in range(1, 9)]
    marker_rows_l_2 = df_left[df_left['Marker'] == 2]

    df_left_final   = df_left.copy()
    df_right_final  = df_right_rough.copy()
    df_angles_final = df_angles.copy()

    if not marker_rows_l_2.empty:
        t_start_fine = marker_rows_l_2.iloc[0]['Time (Seconds)']
        offset_l_fine = calculate_fine_offset_pressure(
            df_left, df_force_rough, cols_Pressure_left, 'Left_Fz',
            t_start_fine, duration=300, fs=target_freq)
        offset_r_fine = calculate_fine_offset_pressure(
            df_right_rough, df_force_rough, cols_Pressure_right, 'Right_Fz',
            t_start_fine, duration=300, fs=target_freq)
        df_left_final['Time (Seconds)']   += offset_l_fine
        df_angles_final['Time (Seconds)'] += offset_l_fine
        df_right_final['Time (Seconds)']  += offset_r_fine

    df_force_final = df_force_rough

    t_start = max(df_left_final['Time (Seconds)'].min(),  df_right_final['Time (Seconds)'].min(),
                  df_angles_final['Time (Seconds)'].min(), df_force_final['Time (Seconds)'].min())
    t_end   = min(df_left_final['Time (Seconds)'].max(),  df_right_final['Time (Seconds)'].max(),
                  df_angles_final['Time (Seconds)'].max(), df_force_final['Time (Seconds)'].max())

    common_time = np.arange(t_start, t_end, 1.0 / target_freq)
    df_merged   = pd.DataFrame({'Time (Seconds)': common_time})

    data_sources = {'L_Dev': df_left_final, 'R_Dev': df_right_final,
                    'Mocap': df_angles_final, 'Force': df_force_final}
    for prefix, df_src in data_sources.items():
        time_col     = 'Time (Seconds)' if 'Time (Seconds)' in df_src.columns else 'Time'
        numeric_cols = df_src.select_dtypes(include=[np.number]).columns
        cols_to_interp = [c for c in numeric_cols if c != time_col and 'Marker' not in c]
        if not cols_to_interp:
            continue
        f = interp1d(df_src[time_col], df_src[cols_to_interp], axis=0,
                     kind='linear', fill_value="extrapolate")
        df_temp = pd.DataFrame(f(common_time), columns=cols_to_interp)
        df_merged = pd.concat([df_merged, df_temp], axis=1)

    df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]

    marker_rows_sync = df_left_final[df_left_final['Marker'] == 2]
    if not marker_rows_sync.empty:
        synced_start = marker_rows_sync.iloc[0]['Time (Seconds)']
        df_analysis  = df_merged[
            (df_merged['Time (Seconds)'] >= synced_start) &
            (df_merged['Time (Seconds)'] <= synced_start + 300.0)
        ].copy()
        df_analysis['Time (Seconds)'] -= synced_start
        return df_analysis.reset_index(drop=True)
    return df_merged

# ============================================================
# 9. ストライド検出・抽出（変更なし）
# ============================================================
def detect_fz_heel_strikes(signal_array, threshold=0.05, min_dist_samples=40):
    is_contact  = signal_array > threshold
    rising_edge = np.diff(is_contact.astype(int), prepend=0) == 1
    potential_indices = np.where(rising_edge)[0]
    if len(potential_indices) == 0:
        return np.array([])
    true_indices = [potential_indices[0]]
    for idx in potential_indices[1:]:
        if idx - true_indices[-1] > min_dist_samples:
            true_indices.append(idx)
    return np.array(true_indices)

def slice_strides_with_constraints(df_input, target_col, side_name="Left",
                                   threshold=0.05, fs=100,
                                   min_duration=0.7, max_duration=1.8):
    sig        = df_input[target_col].values
    time_array = df_input['Time (Seconds)'].values
    hs_indices = detect_fz_heel_strikes(sig, threshold=threshold,
                                        min_dist_samples=int(0.4 * fs))
    valid_strides = []
    for i in range(1, len(hs_indices) - 2):
        start_idx = hs_indices[i]
        end_idx   = hs_indices[i + 1]
        duration  = time_array[end_idx] - time_array[start_idx]
        if min_duration <= duration <= max_duration:
            valid_strides.append(df_input.iloc[start_idx:end_idx].copy())
    print(f"[{side_name}] Accepted Strides: {len(valid_strides)}")
    return valid_strides

# ============================================================
# 10. ストライド時間正規化（変更なし）
# ============================================================
def normalize_strides(stride_list, target_cols, n_points=200):
    normalized_dfs = []
    data_collector = []
    gait_cycle = np.linspace(0, 100, n_points)
    x_new      = np.linspace(0, 1,   n_points)

    for stride_df in stride_list:
        n_len = len(stride_df)
        x_old = np.linspace(0, 1, n_len)
        new_df = pd.DataFrame({'Gait Cycle (%)': gait_cycle})
        stride_matrix = []
        for col in target_cols:
            if col in stride_df.columns:
                y_old = stride_df[col].values
                f     = interp1d(x_old, y_old, kind='linear', fill_value="extrapolate")
                y_new = f(x_new)
            else:
                y_new = np.zeros(n_points)
            new_df[col] = y_new
            stride_matrix.append(y_new)
        normalized_dfs.append(new_df)
        data_collector.append(np.array(stride_matrix).T)

    ensemble_array = (np.array(data_collector) if len(data_collector) > 0
                      else np.empty((0, n_points, len(target_cols))))
    return normalized_dfs, ensemble_array

# ============================================================
# 11. 外れ値ストライド除去
#     変更: std → MAD（外れ値がstd自体を膨らませるのを防ぐ）
# ============================================================
def filter_outlier_strides_mad(ensemble_array, stride_dfs,
                                n_mads=3.5, outlier_ratio_threshold=0.01):
    """
    MAD（Median Absolute Deviation）ベースの外れ値除去．
    median ± n_mads * 1.4826 * MAD で判定．
    1.4826 は正規分布でのstd相当への変換係数．
    """
    if len(ensemble_array) == 0:
        return ensemble_array, stride_dfs, np.array([], dtype=bool)

    median_curve = np.median(ensemble_array, axis=0)      # (200, features)
    mad_curve    = np.median(np.abs(ensemble_array - median_curve[np.newaxis]), axis=0)
    sigma_equiv  = 1.4826 * mad_curve                     # stdと同スケールに変換

    upper = (median_curve + n_mads * sigma_equiv)[np.newaxis]
    lower = (median_curve - n_mads * sigma_equiv)[np.newaxis]

    is_outlier  = (ensemble_array > upper) | (ensemble_array < lower)
    total_pts   = ensemble_array.shape[1] * ensemble_array.shape[2]
    out_ratios  = np.sum(is_outlier, axis=(1, 2)) / total_pts
    keep_mask   = out_ratios <= outlier_ratio_threshold

    clean_ensemble = ensemble_array[keep_mask]
    clean_dfs      = [df for i, df in enumerate(stride_dfs) if keep_mask[i]]

    n_removed = np.sum(~keep_mask)
    if n_removed > 0:
        print(f"  外れ値除去: {n_removed} ストライドを除外 "
              f"({n_removed}/{len(ensemble_array)}, MAD法)")
    return clean_ensemble, clean_dfs, keep_mask

# ============================================================
# 12. 左右データ統合（変更なし）
# ============================================================
def merge_left_right_data(left_ensemble, right_ensemble, left_cols, right_cols):
    right_data = right_ensemble.copy()
    left_data  = left_ensemble.copy()

    flip_cols = {
        'Left_Accel_X': -1,
        'Left_Gyro_Y':  -1,
        'Left_Gyro_Z':  -1,
        'Left_Fx':      -1,
    }
    for col_name, sign in flip_cols.items():
        if col_name in left_cols:
            idx = left_cols.index(col_name)
            left_data[:, :, idx] *= sign

    merged_ensemble = np.concatenate([left_data, right_data], axis=0)
    merged_cols     = [col.replace('Left_', '').replace('Right_', '') for col in left_cols]
    print(f"  左: {left_ensemble.shape[0]}, 右: {right_ensemble.shape[0]}, "
          f"合計: {merged_ensemble.shape[0]} ストライド")
    return merged_ensemble, merged_cols

# ============================================================
# 13. Phase 1: 1被験者1条件のRAW処理（正規化はIMUオフセット除去のみ）
# ============================================================
def process_single_participant_condition_raw(participant, condition, mass):
    """
    生データ → ストライド抽出まで実施．
    グローバル正規化（圧力min-max, IMU std, 関節角度/180）はPhase2で別途適用．
    """
    data     = load_participant_data(participant, condition, mass)
    df_mocap = clean_mocap_columns(data['mocap'])
    df_force = clean_force_columns(data['force'])

    df_left_rs  = process_resampling(data['left'],  sampling_interval=10)
    df_right_rs = process_resampling(data['right'], sampling_interval=10)

    df_left_sm  = process_smoothing_dataframe(df_left_rs,  fs=DEVICE_FREQ)
    df_right_sm = process_smoothing_dataframe(df_right_rs, fs=DEVICE_FREQ)
    df_mocap_sm = process_smoothing_dataframe(df_mocap,    fs=MOCAP_FREQ)
    df_force_sm = process_smoothing_dataframe(df_force,    fs=FORCE_FREQ)

    # 関節角度計算
    df_angles, _, _ = process_mocap_data_target_calibration(df_mocap_sm, df_left_sm)

    # 床反力: 体重正規化 [%BW]（変更なし）
    df_force_bw = normalize_force_by_bodyweight(df_force_sm, mass)

    # IMU: オフセット除去のみ（スケーリングはPhase2）
    # 圧力列名を kPa → Pressure に変換し，負値クリップ（既にfilter後に実施済み）
    df_left_offset  = remove_imu_offset(df_left_sm)
    df_right_offset = remove_imu_offset(df_right_sm)

    # 同期・結合・300秒抽出
    df_final = synchronize_merge_and_extract(
        df_left_offset, df_right_offset, df_angles, df_force_bw, target_freq=100)

    # ストライド抽出
    left_strides  = slice_strides_with_constraints(df_final, 'Left_Fz',  "Left",  threshold=0.05)
    right_strides = slice_strides_with_constraints(df_final, 'Right_Fz', "Right", threshold=0.05)

    # カラム定義（生データ段階での列名）
    cols_left = [
        'Left_Pressure_1', 'Left_Pressure_2', 'Left_Pressure_3', 'Left_Pressure_4',
        'Left_Pressure_5', 'Left_Pressure_6', 'Left_Pressure_7', 'Left_Pressure_8',
        'Left_Accel_X', 'Left_Accel_Y', 'Left_Accel_Z',
        'Left_Gyro_X',  'Left_Gyro_Y',  'Left_Gyro_Z',
        'Left_Hip_Flex_Ext', 'Left_Hip_Abd_Add', 'Left_Hip_Int_Ext_Rot',
        'Left_Knee_Flex_Ext', 'Left_Knee_Abd_Add', 'Left_Knee_Int_Ext_Rot',
        'Left_Ankle_Flex_Ext', 'Left_Ankle_Abd_Add', 'Left_Ankle_Int_Ext_Rot',
        'Left_Fx', 'Left_Fy', 'Left_Fz',
    ]
    cols_right = [c.replace('Left', 'Right') for c in cols_left]

    # ストライド時間正規化（200点）
    left_norm_dfs,  left_ensemble  = normalize_strides(left_strides,  cols_left,  n_points=200)
    right_norm_dfs, right_ensemble = normalize_strides(right_strides, cols_right, n_points=200)

    # 外れ値除去（条件内MAD法）
    L_ens, L_dfs, _ = filter_outlier_strides_mad(left_ensemble,  left_norm_dfs)
    R_ens, R_dfs, _ = filter_outlier_strides_mad(right_ensemble, right_norm_dfs)

    # 左右統合（左足の軸反転含む）
    merged_ensemble, merged_cols = merge_left_right_data(L_ens, R_ens, cols_left, cols_right)

    return {
        'participant':  participant,
        'condition':    condition,
        'mass':         mass,
        'ensemble':     merged_ensemble,   # (N, 200, 26) RAW（グローバル正規化前）
        'columns':      merged_cols,
    }

# ============================================================
# 14. Phase 1: 全データのRAW処理
# ============================================================
def process_all_data_raw(output_dir='raw_strides'):
    """
    全被験者・全条件の生ストライドを抽出して保存．
    グローバル正規化はここでは実施しない．
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    for i, participant in enumerate(PARTICIPANTS):
        mass = MASSES[i]
        for condition in CONDITIONS:
            try:
                result = process_single_participant_condition_raw(participant, condition, mass)
                all_results.append(result)

                # 個別保存（正規化前）
                out_path = os.path.join(output_dir, f"{participant}_{condition}_raw.npz")
                np.savez(out_path,
                         ensemble=result['ensemble'],
                         columns=result['columns'],
                         participant=result['participant'],
                         condition=result['condition'],
                         mass=result['mass'])
                print(f"✓ {participant}_{condition}: {result['ensemble'].shape[0]} strides → {out_path}")

            except Exception as e:
                print(f"✗ エラー: {participant}_{condition}: {e}")
                continue

    return all_results

# ============================================================
# 15. Phase 2: グローバル正規化統計量の計算
#     ※ train/val/test分割後は，train被験者のall_resultsのみ渡して再実行
# ============================================================
def get_column_indices(merged_cols):
    """列名からインデックスを取得するヘルパー"""
    pressure_idx = [i for i, c in enumerate(merged_cols) if 'Pressure' in c]
    imu_idx      = [i for i, c in enumerate(merged_cols) if 'Accel'    in c or 'Gyro' in c]
    angle_idx    = [i for i, c in enumerate(merged_cols)
                    if any(kw in c for kw in ['Hip', 'Knee', 'Ankle'])]
    force_idx    = [i for i, c in enumerate(merged_cols)
                    if c in ['Fx', 'Fy', 'Fz']]
    return pressure_idx, imu_idx, angle_idx, force_idx

def compute_global_stats(all_results, stats_path='normalization_stats.npz'):
    """
    全ストライドデータからグローバル正規化統計量を計算して保存．

    train/val/test分割後の使い方:
        compute_global_stats(train_results_only, 'normalization_stats.npz')
    とすることで，trainのみの統計量に上書き更新できる．

    保存内容:
        pressure_min  : (n_pressure_sensors,) 各センサーの最小値
        pressure_max  : (n_pressure_sensors,) 各センサーの最大値
        imu_std       : (n_imu_axes,)         各IMU軸のstd
        col_indices   : 各モダリティの列インデックス
    """
    print("\n" + "="*60)
    print("グローバル正規化統計量を計算中...")
    merged_cols = all_results[0]['columns']
    pressure_idx, imu_idx, angle_idx, force_idx = get_column_indices(merged_cols)

    all_pressure = []
    all_imu      = []

    for result in all_results:
        ens = result['ensemble']  # (N, 200, features)
        if ens.shape[0] == 0:
            continue
        all_pressure.append(ens[:, :, pressure_idx].reshape(-1, len(pressure_idx)))
        all_imu.append(     ens[:, :, imu_idx     ].reshape(-1, len(imu_idx)))

    all_pressure = np.concatenate(all_pressure, axis=0)  # (N*200, n_pressure)
    all_imu      = np.concatenate(all_imu,      axis=0)  # (N*200, n_imu)

    pressure_min = all_pressure.min(axis=0)
    pressure_max = all_pressure.max(axis=0)
    imu_std      = all_imu.std(axis=0)
    imu_std      = np.where(imu_std < 1e-8, 1.0, imu_std)  # 0除算防止

    np.savez(stats_path,
             pressure_min  = pressure_min,
             pressure_max  = pressure_max,
             imu_std       = imu_std,
             pressure_idx  = np.array(pressure_idx),
             imu_idx       = np.array(imu_idx),
             angle_idx     = np.array(angle_idx),
             force_idx     = np.array(force_idx),
             columns       = np.array(merged_cols))

    print(f"  圧力センサ数: {len(pressure_idx)}")
    print(f"  IMU軸数:      {len(imu_idx)}")
    print(f"  圧力 min: {pressure_min.round(3)}")
    print(f"  圧力 max: {pressure_max.round(3)}")
    print(f"  IMU std:  {imu_std.round(4)}")
    print(f"  → 保存: {stats_path}")
    print("="*60)

    return {'pressure_min': pressure_min, 'pressure_max': pressure_max,
            'imu_std': imu_std, 'pressure_idx': pressure_idx,
            'imu_idx': imu_idx, 'angle_idx': angle_idx, 'force_idx': force_idx}

# ============================================================
# 16. Phase 3: グローバル正規化の適用
# ============================================================
def apply_global_normalization(ensemble, stats):
    """
    ensemble (N, 200, features) にグローバル統計量を適用する．

    適用内容:
        足底圧力 → (x - global_min) / (global_max - global_min)  [0〜1]
        IMU      → x / global_std  （オフセットは既に除去済み）
        関節角度 → x / 180         （固定定数）
        床反力   → そのまま        （既に%BW）
    """
    ens = ensemble.copy()

    pressure_idx = stats['pressure_idx']
    imu_idx      = stats['imu_idx']
    angle_idx    = stats['angle_idx']

    # 足底圧力: グローバルmin-max
    p_min = stats['pressure_min']
    p_max = stats['pressure_max']
    p_range = np.where((p_max - p_min) < 1e-8, 1.0, p_max - p_min)
    ens[:, :, pressure_idx] = (ens[:, :, pressure_idx] - p_min) / p_range

    # IMU: グローバルstdで割る（オフセット除去済み）
    ens[:, :, imu_idx] = ens[:, :, imu_idx] / stats['imu_std']

    # 関節角度: 固定定数 /180
    ens[:, :, angle_idx] = ens[:, :, angle_idx] / ANGLE_SCALE

    return ens

# ============================================================
# 17. Phase 4: 保存
# ============================================================
def save_normalized_dataset(all_results, stats, output_dir='processed_data'):
    """
    正規化済みデータを保存する．
    """
    os.makedirs(output_dir, exist_ok=True)

    unique_participants = sorted(set(r['participant'] for r in all_results))
    participant_map     = {name: i for i, name in enumerate(unique_participants)}
    print(f"\nParticipant ID Map: {participant_map}")

    all_ensembles = []
    all_ids       = []
    all_conditions = []

    condition_map = {'h': 0, 'm': 1, 'l': 2}

    for result in all_results:
        ens_norm = apply_global_normalization(result['ensemble'], stats)

        # 個別ファイル保存
        fname = f"{result['participant']}_{result['condition']}_normalized.npz"
        np.savez(os.path.join(output_dir, fname),
                 ensemble   = ens_norm,
                 columns    = result['columns'],
                 participant = result['participant'],
                 condition   = result['condition'],
                 mass        = result['mass'])

        n_samples = ens_norm.shape[0]
        all_ensembles.append(ens_norm)
        all_ids.append(np.full(n_samples, participant_map[result['participant']]))
        all_conditions.append(np.full(n_samples, condition_map[result['condition']]))

    combined_ensemble  = np.concatenate(all_ensembles,  axis=0)
    combined_ids       = np.concatenate(all_ids,        axis=0)
    combined_conditions = np.concatenate(all_conditions, axis=0)

    combined_path = os.path.join(output_dir, 'all_data_combined.npz')
    np.savez(combined_path,
             ensemble       = combined_ensemble,
             subject_ids    = combined_ids,
             condition_ids  = combined_conditions,
             columns        = all_results[0]['columns'],
             id_map         = participant_map,
             condition_map  = condition_map,
             angle_scale    = ANGLE_SCALE)  # 逆変換用

    print(f"\n統合データ保存完了: {combined_path}")
    print(f"  ensemble shape:      {combined_ensemble.shape}")
    print(f"  subject_ids shape:   {combined_ids.shape}")
    print(f"  condition_ids shape: {combined_conditions.shape}")
    print(f"\n  ※ 関節角度の逆変換: pred_angle_deg = pred_normalized * {ANGLE_SCALE}")
    print(f"  ※ 正規化統計量は normalization_stats.npz に保存済み")
    print(f"\n  ★ train/val/test分割後は以下を実行して統計量を更新してください:")
    print(f"     compute_global_stats(train_results, 'normalization_stats.npz')")
    print(f"     → その後 apply_global_normalization() を全splitに適用")

# ============================================================
# メイン実行
# ============================================================
if __name__ == "__main__":
    # --- Phase 1: 全データの生ストライド抽出 ---
    print("\n【Phase 1】全データの生ストライド抽出 (グローバル正規化前)")
    all_results = process_all_data_raw(output_dir='raw_strides')

    if len(all_results) == 0:
        print("処理できたデータがありません．終了します．")
        exit(1)

    # --- Phase 2: グローバル統計量の計算（現時点では全データ） ---
    print("\n【Phase 2】グローバル正規化統計量の計算")
    print("  ※ 現時点では全データから計算．train/val/test分割後はtrainのみで再計算すること")
    stats = compute_global_stats(all_results, stats_path='normalization_stats.npz')

    # --- Phase 3 & 4: 正規化適用 → 保存 ---
    print("\n【Phase 3 & 4】正規化の適用と保存")
    save_normalized_dataset(all_results, stats, output_dir='processed_data')

    print("\n" + "="*60)
    print("全処理が完了しました")
    print("="*60)