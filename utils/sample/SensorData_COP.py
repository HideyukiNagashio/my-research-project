import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 実験者のリスト
participants = [
    "tadane", "nagashio", "fujii", "kiuchi", "nagayoshi", 
    "kanda", "suetake", "ono", "son", "yanaze"
]

# 各実験者の体重（kg）
weights = {
    "tadane": 53.8, "nagashio": 77.5, "fujii": 80.3, "kiuchi": 69.7, "nagayoshi": 73.9,
    "kanda": 59.8, "suetake": 65.6, "ono": 59.6, "son": 83.6, "yanaze": 56.8
}

# データフォルダのパス
data_folder = r'C:\WorkSpace\B_analysis\SensorData'
output_folder = r'C:\WorkSpace\B_analysis\ProcessedData'
os.makedirs(output_folder, exist_ok=True)

# ローパスフィルタの設計
def butter_lowpass(cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# データにローパスフィルタを適用
def apply_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, data)

# COP計算用関数
def calculate_COP(df, coefficients, positions):
    COPx = 0
    COPy = 0
    total_force = 0
    for foot, (x_pos, y_pos) in positions.items():
        force = df[foot] * coefficients[foot]
        COPx += force * x_pos
        COPy += force * y_pos
        total_force += force
    COPx /= total_force
    COPy /= total_force
    return COPx, COPy

# 定数
fs = 100  # サンプリング周波数
cutoff = 25  # ローパスフィルタのカットオフ周波数
R = 180
D = 9.5 / 1000  # mmからmに変換
area = np.pi * (D / 2)**2

coefficients = {
    "Right_Foot_1": 0.22497882, "Right_Foot_2": 4.219779047, "Right_Foot_3": 4.309939138,
    "Right_Foot_4": 4.593853313, "Right_Foot_5": 4.248982725, "Right_Foot_6": 4.973555078,
    "Right_Foot_7": 3.462631472, "Right_Foot_8": 4.772374839,
    "Left_Foot_1": 3.966326901, "Left_Foot_2": 4.744929171, "Left_Foot_3": 4.15168471,
    "Left_Foot_4": 4.56279606, "Left_Foot_5": 4.079466017, "Left_Foot_6": 3.484332969,
    "Left_Foot_7": 4.341571247, "Left_Foot_8": 5.130518067,
}
positions = {
    "Right_Foot_1": (27, 218), "Right_Foot_2": (23, 183), "Right_Foot_3": (46, 182),
    "Right_Foot_4": (74, 176), "Right_Foot_5": (74, 111), "Right_Foot_6": (64, 52),
    "Right_Foot_7": (38, 44), "Right_Foot_8": (50, 19),
    "Left_Foot_1": (-27, 218), "Left_Foot_2": (-23, 183), "Left_Foot_3": (-46, 182),
    "Left_Foot_4": (-74, 176), "Left_Foot_5": (-74, 111), "Left_Foot_6": (-64, 52),
    "Left_Foot_7": (-38, 44), "Left_Foot_8": (-50, 19),
}

# 各実験者のフォルダを処理
for participant in participants:
    participant_folder = os.path.join(data_folder, participant)
    if not os.path.exists(participant_folder):
        print(f"フォルダが存在しません: {participant_folder}")
        continue

    # フォルダ内の各速度について処理
    for speed in ["f", "n", "s"]:
        left_file_name = f"{participant}_{speed}_left_foot_data.csv"
        right_file_name = f"{participant}_{speed}_right_foot_data.csv"

        left_file_path = os.path.join(participant_folder, left_file_name)
        right_file_path = os.path.join(participant_folder, right_file_name)

        if not os.path.exists(left_file_path) or not os.path.exists(right_file_path):
            print(f"ファイルが見つかりません: {left_file_path} または {right_file_path}")
            continue

        # 左右のデータを読み込み
        df_left = pd.read_csv(left_file_path, header=0)
        df_right = pd.read_csv(right_file_path, header=0)

        # ElapsedTime以外の値が10以下なら0に設定
        df_left[df_left.columns.difference(["ElapsedTime"])] = df_left[df_left.columns.difference(["ElapsedTime"])].applymap(lambda x: 0 if x <= 10 else x)
        df_right[df_right.columns.difference(["ElapsedTime"])] = df_right[df_right.columns.difference(["ElapsedTime"])].applymap(lambda x: 0 if x <= 10 else x)

        # ローパスフィルタを適用
        for column in df_left.columns:
            if column != 'ElapsedTime':
                df_left[column] = apply_lowpass_filter(df_left[column], cutoff, fs)
                df_left[column] = np.maximum(df_left[column], 0)  # 0未満の値を0に置き換え

        for column in df_right.columns:
            if column != 'ElapsedTime':
                df_right[column] = apply_lowpass_filter(df_right[column], cutoff, fs)
                df_right[column] = np.maximum(df_right[column], 0)  # 0未満の値を0に置き換え

        # 左右データをマージ
        df_merged = pd.merge(df_left, df_right, on='ElapsedTime', how='outer')
        df_merged = df_merged.interpolate(method='linear', axis=0)

        # 力の計算
        for column in df_merged.columns:
            if column != "ElapsedTime":
                df_merged[column] = 1000 * df_merged[column] / ((1023 - df_merged[column]) * R)

        # 圧力の計算
        for col, coeff in coefficients.items():
            if col in df_merged.columns:
                df_merged[col] = df_merged[col] * coeff # N
                df_merged[col] = df_merged[col] / area / 1000 # kPa

        # COP計算
        df_merged["COPx"], df_merged["COPy"] = zip(*df_merged.apply(lambda row: calculate_COP(row, coefficients, positions), axis=1))

        # 正規化
        weight = weights.get(participant, None)
        if weight is not None:
            for col in df_merged.columns:
                if col not in ["ElapsedTime", "COPx", "COPy"]:
                    df_merged[col] = df_merged[col] / weight
        else:
            print(f"体重情報が見つかりません: {participant}")

        # 正規化後のRightおよびLeftの合計値を計算し、新しいカラムを追加
        df_merged["Right_Total"] = df_merged[[
            "Right_Foot_1", "Right_Foot_2", "Right_Foot_3", "Right_Foot_4", "Right_Foot_5", 
            "Right_Foot_6", "Right_Foot_7", "Right_Foot_8"]].sum(axis=1)

        df_merged["Left_Total"] = df_merged[[
            "Left_Foot_1", "Left_Foot_2", "Left_Foot_3", "Left_Foot_4", "Left_Foot_5", 
            "Left_Foot_6", "Left_Foot_7", "Left_Foot_8"]].sum(axis=1)
        
        # 時間軸の補完
        time_min = df_merged['ElapsedTime'].min()
        time_max = df_merged['ElapsedTime'].max()

        # 100Hzの間隔で時間を生成
        new_time = np.arange(time_min, time_max, 10)  # 0.01秒間隔（100Hz）

        # 各列の補完（線形補完）
        interpolated_data = pd.DataFrame()
        interpolated_data['ElapsedTime'] = new_time

        for column in df_merged.columns[1:]:  # ElapsedTime以外の列
            interpolator = interp1d(df_merged['ElapsedTime'], df_merged[column], kind='linear', fill_value='extrapolate')
            interpolated_data[column] = interpolator(new_time)

        # 結果を保存
        output_file_name = f"{participant}_{speed}_merged_processed_data.csv"
        output_path = os.path.join(output_folder, output_file_name)
        interpolated_data.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"処理完了: {output_path}")
