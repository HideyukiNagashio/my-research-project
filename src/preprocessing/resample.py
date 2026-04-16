import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

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
