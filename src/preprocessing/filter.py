from scipy import signal
from .config import CUTOFF_FREQ, FILTER_ORDER, DEVICE_FREQ

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

def remove_imu_offset(df_input):
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
