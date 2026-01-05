# processor.py

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from collections import deque
import time
from pathlib import Path
from typing import Optional

from gait_feedback.grf import GRFReader
from gait_feedback.config import (
    JOINT_NO, TIME_SERIES_LEN, SAMPLING,
    MAX_ARR, MIN_ARR, CONTACT_RATE,
    R_TOE_INDEX,        
    TOE_HEIGHT_MAX, TOE_HEIGHT_MIN,  
)
from gait_feedback.angle import CalculateAngle

class GaitProcessor:
    """
    歩容データの取得、前処理まで、
    データに関する一連の処理をすべて担当するクラス。
    """
    def __init__(self, classify_model, generator_model):
        self.classify_model = classify_model
        self.generator_model = generator_model
        self.angle_calculator = CalculateAngle()

    # ====================================================
    #  リアルタイムストライド取得
    # ====================================================
    def capture_stride(self, wrapper, weight, basic_angle, pbar, grf_reader: GRFReader):
        """ノイズ除去を考慮し、リアルタイムで1ストライドを切り出す"""
        pbar.set_description("[Waiting for stride start]")
        pbar.refresh()

        SWING_FRAMES_FOR_DEBOUNCE = 10
        r_state_history = deque(maxlen=SWING_FRAMES_FOR_DEBOUNCE)
        pos_buffer, grf_buffer = [], []
        stride_started = False

        # 計測用メタ情報
        t_stride_start = None
        frames_recorded = 0
        stride_meta = dict(
            duration_sec=float("nan"),
            frames=0,
            fps=float("nan"),
            ms_per_frame=float("nan"),
        )

        while True:
            # Motive から最新の剛体座標を取得
            _, pos = wrapper.get_latest_positions()
            # GRF 1 サンプル取得
            current_grf_raw = grf_reader.read_single_sample()

            if pos.size == 0:
                time.sleep(0.001)
                continue

            # --- ストライド判定ロジック ---
            g = current_grf_raw[:, 2] * 100 / (weight * 9.8)  # [BW] 換算
            now_R = "st" if g[0] > CONTACT_RATE else "sw"

            is_clean_contact = (
                now_R == "st"
                and len(r_state_history) == SWING_FRAMES_FOR_DEBOUNCE
                and all(s == "sw" for s in r_state_history)
            )
            r_state_history.append(now_R)

            if not stride_started:
                if is_clean_contact:
                    print("\n[INFO] Stride start detected. Recording...")
                    stride_started = True
                    pos_buffer = [pos]
                    grf_buffer = [current_grf_raw]
                    frames_recorded = 1
                    t_stride_start = time.perf_counter()
            else:
                pos_buffer.append(pos)
                grf_buffer.append(current_grf_raw)
                frames_recorded += 1

                pbar.set_description(f"[Recording Stride... Frame: {len(pos_buffer)}]")
                pbar.refresh()

                if is_clean_contact:
                    # ストライド終端
                    t_elapsed = (
                        time.perf_counter() - t_stride_start
                        if t_stride_start is not None
                        else float("nan")
                    )
                    fps = (
                        frames_recorded / t_elapsed
                        if (t_elapsed > 0 and np.isfinite(t_elapsed))
                        else float("nan")
                    )
                    ms_per_frame = (
                        1000.0 / fps
                        if (fps > 0 and np.isfinite(fps))
                        else float("nan")
                    )

                    print("\n[INFO] Stride end detected. Capture finished.")
                    print("------ Stride Capture Stats ------")
                    print(f"Frames: {frames_recorded}")
                    print(f"Time  : {t_elapsed:.3f} s")
                    print(f"FPS   : {fps:.2f} frames/s")
                    print(f"Δt    : {ms_per_frame:.2f} ms/frame")
                    print("----------------------------------")

                    stride_meta = dict(
                        duration_sec=round(float(t_elapsed), 6)
                        if np.isfinite(t_elapsed)
                        else None,
                        frames=int(frames_recorded),
                        fps=round(float(fps), 3) if np.isfinite(fps) else None,
                        ms_per_frame=round(float(ms_per_frame), 3)
                        if np.isfinite(ms_per_frame)
                        else None,
                    )

                    pbar.set_description("Training")
                    pbar.refresh()
                    break

            time.sleep(0.001)

        # pos: (rigid, frame, 3), grf: (plate, frame, 3)
        return (
            np.asarray(pos_buffer).transpose(1, 0, 2),
            np.asarray(grf_buffer).transpose(1, 0, 2),
            stride_meta,
        )

    # ====================================================
    #  前処理系（共通で使用）
    # ====================================================
    def _interpolation(self, arr: np.ndarray) -> np.ndarray:
        """各チャンネル・各軸を3次スプラインで補間"""
        out = np.zeros_like(arr)
        for ch in range(arr.shape[0]):
            for axis in range(arr.shape[2]):
                out[ch, :, axis] = (
                    pd.Series(arr[ch, :, axis]).interpolate("cubic").bfill().values
                )
        return out

    def _process_raw_stride_data(
        self,
        pos_np,
        grf_np,
        basic_angle,
        weight,
        base_r_toe_mm: float | None = None,   # ★ 追加
    ):

        """
        【共通の心臓部】
        - 位置座標・床反力を補間
        - angle.py(CalculateAngle) を用いて3D関節角度を算出
        - 基準姿勢との差分角度を取得
        - GRF を体重で正規化し 6 変量にまとめる
        - R_TOE_INDEX の Z 座標(母趾床間距離; mm換算)の波形も返す
        """
        # 補間
        pos_np_interpolated = self._interpolation(pos_np)  # (rigid, frame, 3)
        grf_np_interpolated = self._interpolation(grf_np)  # (plate, frame, 3)

        # 関節角度計算（angle.py を使用）
        num_frames = pos_np_interpolated.shape[1]
        ang = np.empty((JOINT_NO, num_frames, 3), dtype=np.float32)
        for t in range(num_frames):
            # CalculateAngle.angles: (rigid,3) -> (JOINT_NO,3)
            ang[:, t, :] = self.angle_calculator.angles(
                pos_np_interpolated[:, t, :]
            )

        # 基準姿勢との差分角度（無ければそのまま）
        diff = ang - basic_angle if basic_angle is not None else ang
        # yz / zx 成分のみ使用 → (7, T, 2)
        angle_diff_data = diff[:, :, [1, 2]]

        # 床反力を体重で正規化して 6 変量にまとめる
        grf_n_raw = grf_np_interpolated / weight
        grf6_raw = np.stack(
            [
                -grf_n_raw[0, :, 0],
                grf_n_raw[0, :, 1],
                grf_n_raw[0, :, 2],
                -grf_n_raw[2, :, 0],
                grf_n_raw[2, :, 1],
                grf_n_raw[2, :, 2],
            ]
        )

        # ★ R拇指床間距離 (Z軸, m → mm) の生波形 (frames,)
        toe_mm_abs = pos_np_interpolated[R_TOE_INDEX - 1, :, 1] * 1000.0  # 絶対値 [mm]

        if base_r_toe_mm is not None:
            # 他の変量と同じく「基本姿勢との差分」を使う
            toe_mm_raw = toe_mm_abs - float(base_r_toe_mm)
        else:
            # 後方互換のため、指定がなければ従来通り絶対値
            toe_mm_raw = toe_mm_abs


        return angle_diff_data, grf6_raw, toe_mm_raw  # (7,T,2), (6,T), (T,)

    def _resize_separated(self, angle_data_raw, grf_data_raw):
        """分離されたデータ（角度と床反力）をそれぞれリサイズする"""
        original_len = angle_data_raw.shape[1]
        new_len = TIME_SERIES_LEN

        # 角度: (7, T, 2) -> (7, 400, 2)
        resized_angle = np.zeros((7, new_len, 2), dtype=np.float32)
        for joint in range(7):
            for plane in range(2):
                original_series = angle_data_raw[joint, :, plane]
                new_x = np.linspace(0, original_len - 1, new_len)
                original_x = np.arange(original_len)
                resized_angle[joint, :, plane] = np.interp(
                    new_x, original_x, original_series
                )

        # GRF: (6, T) -> (6, 400)
        resized_grf = np.zeros((6, new_len), dtype=np.float32)
        for var in range(6):
            original_series = grf_data_raw[var, :]
            new_x = np.linspace(0, original_len - 1, new_len)
            original_x = np.arange(original_len)
            resized_grf[var, :] = np.interp(new_x, original_x, original_series)

        return resized_angle, resized_grf

    def _normalize_separated(self, angle_data, grf_data):
        """分離されたデータ（角度と床反力）をそれぞれ正規化する"""
        normalized_angle = angle_data.copy()
        var_index_counter = 0

        # まず 14 変量（7関節×2平面）
        for joint_idx in range(normalized_angle.shape[0]):
            for plane_idx in range(normalized_angle.shape[2]):
                series = normalized_angle[joint_idx, :, plane_idx]
                denominator = (
                    MAX_ARR[var_index_counter] - MIN_ARR[var_index_counter]
                )
                if denominator == 0:
                    denominator = 1
                normalized_series = (
                    series - MIN_ARR[var_index_counter]
                ) / denominator
                normalized_angle[joint_idx, :, plane_idx] = normalized_series
                var_index_counter += 1

        # GRF 6 変量（インデックス 14〜19）
        normalized_grf = grf_data.copy()
        for i in range(6):
            var_index = 14 + i
            denominator = MAX_ARR[var_index] - MIN_ARR[var_index]
            if denominator == 0:
                denominator = 1
            normalized_grf[i, :] = (
                normalized_grf[i, :] - MIN_ARR[var_index]
            ) / denominator

        # ★★ ここを削除（clip しない）★★
        # normalized_angle = np.clip(normalized_angle, 0.0, 1.0)
        # normalized_grf = np.clip(normalized_grf, 0.0, 1.0)

        return normalized_angle, normalized_grf


    def _lowpass_separated(self, angle_data, grf_data, fc=10.0, order=4):
        """分離されたデータにそれぞれローパスフィルタをかける"""
        fs = SAMPLING
        b, a = butter(order, fc / (fs * 0.5), btype="low", analog=False)

        filtered_angle = np.zeros_like(angle_data)
        for joint in range(7):
            for plane in range(2):
                filtered_angle[joint, :, plane] = filtfilt(
                    b, a, angle_data[joint, :, plane]
                )

        filtered_grf = np.zeros_like(grf_data)
        for i in range(6):
            filtered_grf[i, :] = filtfilt(b, a, grf_data[i, :])

        return filtered_angle, filtered_grf

    def _lowpass_1d(self, series, fc=10.0, order=4):
        """1次元波形用のローパスフィルタ（toe 用）"""
        fs = SAMPLING
        b, a = butter(order, fc / (fs * 0.5), btype="low", analog=False)
        return filtfilt(b, a, series)

    # ====================================================
    #  デバッグ用: 角度(7,T,2) & GRF(6,T) を CSV 出力
    # ====================================================
    def _dump_debug_angle_grf(self, angle_7xTx2, grf_6xT, fname: str):
        """
        angle_7xTx2: (7, T, 2)
        grf_6xT    : (6, T)
        fname      : 出力ファイル名（debug_waveforms/以下）
        """
        debug_dir = Path("debug_waveforms")
        debug_dir.mkdir(exist_ok=True)

        T = angle_7xTx2.shape[1]

        # 角度 (7×2=14変量) を (T,14) に並べる
        angle_cols = []
        angle_series = []
        for joint in range(angle_7xTx2.shape[0]):
            for plane in range(angle_7xTx2.shape[2]):
                angle_cols.append(f"j{joint}_p{plane}")
                angle_series.append(angle_7xTx2[joint, :, plane])
        angle_arr = np.stack(angle_series, axis=1)  # (T,14)

        # GRF (6,T) → (T,6)
        grf_arr = grf_6xT.T  # (T,6)
        grf_cols = [f"grf{i}" for i in range(grf_arr.shape[1])]

        # 結合 (T, 20)
        all_arr = np.concatenate([angle_arr, grf_arr], axis=1)
        all_cols = angle_cols + grf_cols

        df = pd.DataFrame(all_arr, columns=all_cols)
        df.to_csv(debug_dir / fname, index=False)

    # ====================================================
    #  AI 解析（分類＋生成）本体
    # ====================================================
    def analyze(self, data):
        """
        data: (1,20,400,1) 正規化済みテンソル
        → cls (0 or 1), target_tensor(1,20,400,1) を返す
        """
        x = data.squeeze(-1).transpose(0, 2, 1)  # (1,400,20)
        preds = self.classify_model.predict(x, verbose=0)

        # softmax: [つまずきやすい, つまずきにくい]
        p_risky = float(preds[0][0])  # cls=0: つまずきやすい
        p_safe = float(preds[0][1])   # cls=1: つまずきにくい
        cls = int(np.argmax(preds))

        print(
            f"[DEBUG] preds={preds[0]}  -> "
            f"p_risky={p_risky:.3f}, p_safe={p_safe:.3f}, "
            f"cls={cls} ({'risky' if cls == 0 else 'safe'})"
        )

        # 危険な歩容（つまずきやすい）= cls==0 のときに生成を行う
        if cls == 0:
            gen = self.generator_model.predict(x, verbose=0)
            tgt = gen.transpose(0, 2, 1)[..., np.newaxis]
        else:
            # つまずきにくい歩容なら、そのまま返す
            tgt = data.copy()

        return cls, tgt

    # ====================================================
    #  「前処理だけ」版（30秒平均用）
    # ====================================================
    def preprocess_stride_only(
        self,
        pos_np,
        grf_np,
        basic_angle,
        weight,
        base_r_toe_mm: float | None = None,   # ★ 追加
        debug_prefix: Optional[str] = None,
    ):

        """
        生データ(pos_np, grf_np)から
        ・角度/床反力計算（angle.py 経由）
        ・R拇指床間距離(Z; mm)の計算
        ・リサイズ(400点)
        ・正規化(0..1)  ※toe は TOE_HEIGHT_MIN/MAX を使用
        ・ローパス
        まで行い、
        (20, TIME_SERIES_LEN), toe(400,) を返す。
        分類・生成は行わない。

        debug_prefix を与えると、処理ステップごとに debug_waveforms/ 配下へ
        CSV を出力する。
        """
        angle_data_raw, grf_data_raw, toe_mm_raw = self._process_raw_stride_data(
            pos_np, grf_np, basic_angle, weight,
            base_r_toe_mm=base_r_toe_mm,   # ★ 差分 toe 用
        )


        # ★ デバッグ: 生の角度差分 & GRF6
        if debug_prefix is not None:
            self._dump_debug_angle_grf(
                angle_data_raw,
                grf_data_raw,
                fname=f"{debug_prefix}_01_raw_angle_grf.csv",
            )

        # --- 角度・GRF を従来通り処理 ---
        angle_resized, grf_resized = self._resize_separated(
            angle_data_raw, grf_data_raw
        )
        if debug_prefix is not None:
            self._dump_debug_angle_grf(
                angle_resized,
                grf_resized,
                fname=f"{debug_prefix}_02_resized_angle_grf.csv",
            )

        angle_norm, grf_norm = self._normalize_separated(
            angle_resized, grf_resized
        )
        if debug_prefix is not None:
            self._dump_debug_angle_grf(
                angle_norm,
                grf_norm,
                fname=f"{debug_prefix}_03_norm_angle_grf.csv",
            )

        angle_filtered, grf_filtered = self._lowpass_separated(
            angle_norm, grf_norm
        )
        if debug_prefix is not None:
            self._dump_debug_angle_grf(
                angle_filtered,
                grf_filtered,
                fname=f"{debug_prefix}_04_filt_angle_grf.csv",
            )

        num_timesteps = angle_filtered.shape[1]
        data_for_ai = np.zeros((20, num_timesteps), dtype=np.float32)

        var_counter = 0
        # 7 関節 × 2 平面 = 14 変量
        for i in range(7):
            for j in range(2):
                data_for_ai[var_counter, :] = angle_filtered[i, :, j]
                var_counter += 1

        # GRF 6 変量
        for i in range(6):
            data_for_ai[var_counter, :] = grf_filtered[i, :]
            var_counter += 1

        # ★ 最終 20×400 も保存
        if debug_prefix is not None:
            debug_dir = Path("debug_waveforms")
            debug_dir.mkdir(exist_ok=True)
            df_final = pd.DataFrame(data_for_ai.T)  # (400,20)
            df_final.to_csv(
                debug_dir / f"{debug_prefix}_05_final_20x400.csv",
                index=False,
            )

        # --- R拇指床間距離 Toe(mm) を 400点にリサンプル＆正規化 ---
        original_len = toe_mm_raw.shape[0]
        original_x = np.arange(original_len)
        new_x = np.linspace(0, original_len - 1, TIME_SERIES_LEN)
        toe_mm_resized = np.interp(new_x, original_x, toe_mm_raw)

        denom = TOE_HEIGHT_MAX - TOE_HEIGHT_MIN
        if denom == 0:
            denom = 1.0

        # 0〜1に押し込まず、そのまま正規化してフィルタだけかける
        toe_norm = (toe_mm_resized - TOE_HEIGHT_MIN) / denom
        toe_norm = self._lowpass_1d(toe_norm)


        if debug_prefix is not None:
            debug_dir = Path("debug_waveforms")
            debug_dir.mkdir(exist_ok=True)
            df_toe = pd.DataFrame(
                {
                    "toe_norm": toe_norm,
                    "toe_mm_resized": toe_mm_resized,
                }
            )
            df_toe.to_csv(
                debug_dir / f"{debug_prefix}_06_toe.csv",
                index=False,
            )

        return data_for_ai, toe_norm  # (20,400), (400,)

    # ====================================================
    #  オフライン一括処理用（必要なら使用）
    # ====================================================
    def process_stride_data(self, pos_np, grf_np, basic_angle, weight):
        """
        生データから AI 解析まで実行し、
        cls, original_processed(20,400), target_processed(20,400) を返す。
        """
        angle_data_raw, grf_data_raw, toe_mm_raw = self._process_raw_stride_data(
            pos_np, grf_np, basic_angle, weight
        )

        angle_resized, grf_resized = self._resize_separated(
            angle_data_raw, grf_data_raw
        )
        angle_norm, grf_norm = self._normalize_separated(
            angle_resized, grf_resized
        )
        angle_filtered, grf_filtered = self._lowpass_separated(
            angle_norm, grf_norm
        )

        num_timesteps = angle_filtered.shape[1]
        data_for_ai = np.zeros((20, num_timesteps), dtype=np.float32)

        var_counter = 0
        for i in range(7):
            for j in range(2):
                data_for_ai[var_counter, :] = angle_filtered[i, :, j]
                var_counter += 1
        for i in range(6):
            data_for_ai[var_counter, :] = grf_filtered[i, :]
            var_counter += 1

        final_data_tensor = data_for_ai[np.newaxis, ..., np.newaxis]
        cls, target_tensor = self.analyze(final_data_tensor)

        original_processed_data = final_data_tensor[0, :, :, 0]
        target_processed_data = target_tensor[0, :, :, 0]

        return cls, original_processed_data, target_processed_data
