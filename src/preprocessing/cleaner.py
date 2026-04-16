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
