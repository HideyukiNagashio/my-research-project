import numpy as np
from .config import ANGLE_SCALE

def normalize_force_by_bodyweight(df_force, mass):
    cols  = ['Time (Seconds)', 'Right_Fx', 'Right_Fy', 'Right_Fz',
             'Left_Fx', 'Left_Fy', 'Left_Fz']
    fcols = [c for c in cols if c != 'Time (Seconds)']
    df    = df_force[cols].copy()
    df[fcols] /= mass * 9.81
    return df

def _col_indices(cols):
    p_idx  = [i for i, c in enumerate(cols) if 'Pressure' in c and 'Contra' not in c]
    i_idx  = [i for i, c in enumerate(cols) if ('Accel' in c or 'Gyro' in c) and 'Contra' not in c]
    a_idx  = [i for i, c in enumerate(cols) if any(k in c for k in ('Hip','Knee','Ankle')) and 'Contra' not in c]
    cp_idx = [i for i, c in enumerate(cols) if 'Pressure' in c and 'Contra' in c]
    ci_idx = [i for i, c in enumerate(cols) if ('Accel' in c or 'Gyro' in c) and 'Contra' in c]
    ca_idx = [i for i, c in enumerate(cols) if any(k in c for k in ('Hip','Knee','Ankle')) and 'Contra' in c]
    return p_idx, i_idx, a_idx, cp_idx, ci_idx, ca_idx

def compute_global_stats(train_results, stats_path=None):
    """
    グローバル正規化統計量を計算する．
    交差検証（CV）時に test データが漏洩しないよう，train_results（学習用subset）のみを入力すること．
    
    【足底圧力】
    ipsi + contra を合わせた全センサ・全データの単一スカラーmax を使用する．
    
    【IMU】
    ipsi 側の軸ごと std を計算し，contra にも同じ値を適用する（同一物理量のため）．
    """
    print("\n【Normalization】学習用データからグローバル統計量を計算中...")
    if not train_results:
        raise ValueError("train_results が空です")
        
    cols = train_results[0]['columns']
    p_idx, i_idx, a_idx, cp_idx, ci_idx, ca_idx = _col_indices(cols)

    all_p, all_i = [], []
    for r in train_results:
        ens = r['ensemble']
        if ens.shape[0] == 0:
            continue
        p_combined = ens[:, :, p_idx + cp_idx].reshape(-1, len(p_idx) + len(cp_idx))
        i_ipsi     = ens[:, :, i_idx         ].reshape(-1, len(i_idx))
        all_p.append(p_combined)
        all_i.append(i_ipsi)

    if not all_p:
        raise ValueError("有効なストライドデータが含まれていません")

    all_p = np.concatenate(all_p, axis=0)
    all_i = np.concatenate(all_i, axis=0)

    p_global_max = float(all_p.max())
    p_global_min = 0.0

    i_std = all_i.std(axis=0)
    i_std = np.where(i_std < 1e-8, 1.0, i_std)

    stats = dict(pressure_global_max=p_global_max, pressure_global_min=p_global_min,
                 imu_std=i_std,
                 p_idx=p_idx,   i_idx=i_idx,   a_idx=a_idx,
                 cp_idx=cp_idx, ci_idx=ci_idx, ca_idx=ca_idx)

    if stats_path:
        np.savez(stats_path,
                 pressure_global_max = np.array(p_global_max),
                 pressure_global_min = np.array(p_global_min),
                 imu_std             = i_std,
                 p_idx  = np.array(p_idx),  i_idx  = np.array(i_idx),  a_idx  = np.array(a_idx),
                 cp_idx = np.array(cp_idx), ci_idx = np.array(ci_idx), ca_idx = np.array(ca_idx),
                 columns = np.array(cols))
        print(f"  保存: {stats_path}")

    print(f"  圧力 global_max (ipsi+contra 合算): {p_global_max:.6f}")
    print(f"  IMU std 6軸 (ipsi 基準):            {i_std.round(4)}")

    return stats

def apply_global_normalization(ensemble, stats):
    """
    52ch ensemble にグローバル統計量を適用する．
    引数 stats は compute_global_stats で計算されたもの．
    """
    ens   = ensemble.copy()
    p_max = float(stats['pressure_global_max'])
    p_min = float(stats['pressure_global_min'])
    p_range = max(p_max - p_min, 1e-8)
    i_std   = stats['imu_std']

    for idx in [stats['p_idx'], stats['cp_idx']]:
        ens[:, :, idx] = np.clip((ens[:, :, idx] - p_min) / p_range, 0.0, 1.0)

    ens[:, :, stats['i_idx']]  /= i_std
    ens[:, :, stats['ci_idx']] /= i_std

    for idx in [stats['a_idx'], stats['ca_idx']]:
        ens[:, :, idx] /= ANGLE_SCALE

    return ens
