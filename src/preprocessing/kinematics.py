import numpy as np
import pandas as pd
from numpy import linalg as LA
from .config import RI, RGT, RK, RM, RT, LI, LGT, LK, LM, LT, JOINT_NO, SIGN_TABLE

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
