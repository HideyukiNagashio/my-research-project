# config.py

from pathlib import Path
import numpy as np
from win32api import GetSystemMetrics

ROOT = Path(__file__).resolve().parents[2]

# data root
DATA_ROOT   = ROOT
TARGET_DIR  = DATA_ROOT / "target"
REALTIME_ROOT = DATA_ROOT / "data"

# --- Unity 連携用フォルダ ---
UNITY_INBOX    = TARGET_DIR
UNITY_INBOX.mkdir(parents=True, exist_ok=True)
UNITY_MANIFEST = UNITY_INBOX / "LATEST.json"

# --- 訓練前後保存先 ---
NON_FEEDBACK_BEFORE_DIR = REALTIME_ROOT / "beforeafter"
NON_FEEDBACK_BEFORE_DIR.mkdir(parents=True, exist_ok=True)

NON_FEEDBACK_FEEDBACKTIME_DIR = REALTIME_ROOT / "feedbacktime"
NON_FEEDBACK_FEEDBACKTIME_DIR.mkdir(parents=True, exist_ok=True)

# --- 画像 / モデル ---
GAIT_IMG_PATH       = ROOT / "information"   / "white.png"
CLASSIFY_MODEL_PATH = ROOT / "models" / "251114best_model2.h5"
GENERATOR_MODEL_PATH= ROOT / "models" / "best_cyclegan_model"


# ============================================================
#  共通パラメータ（時間・ストライド・サンプリング）
# ============================================================

TRAINING_TIME   = 600       # [s] 訓練時間
UPDATE_INTERVAL = 30        # [s] 30秒ごとのフィードバック更新
RIGIDBODY_NO    = 13        # 使用する剛体数

# ベースライン(フィードバック無し60秒)
BASELINE_TIME        = 60.0  # [s] ベースライン全体
BASELINE_IGNORE_HEAD = 15.0  # [s] 最初の15秒は解析に使わない
BASELINE_AVG_WINDOW  = 30.0  # [s] その後30秒分で平均 (15～45秒)

# 歩容データ仕様
JOINT_NO        = 7          # 使用関節数（角度計算で使用）
SAMPLING        = 250        # [Hz] サンプリング周波数
TIME_SERIES_LEN = 400        # 1ストライドあたりサンプル数

CONTACT_RATE = 10            # 床反力の閾値 [%]
GUI_SCALE    = 0.55          # 画面縮小率

# 拇趾床間距離（正規化前の物理量レンジ）
TOE_HEIGHT_MAX = 148.32
TOE_HEIGHT_MIN = -23.46


# ============================================================
#  トレッドミル制御パラメータ
# ============================================================

TREADMILL_ENABLED        = True        # マスターON/OFFスイッチ
TREADMILL_DEVICE         = "Dev1"      # NI-DAQデバイス名
TREADMILL_AO_CHANNEL     = "ao0"       # アナログ出力チャンネル
TREADMILL_TARGET_RB_ID   = 3           # 制御ターゲットの剛体ID

# 制御ゲイン (PID)
TREADMILL_KP = 8.0   # 比例ゲイン
TREADMILL_KI = 0.5   # 積分ゲイン
TREADMILL_KD = 0.2   # 微分ゲイン

# 制御ロジック
TREADMILL_CENTER_DEADBAND_M = 0.04   # 制御しない中心領域 (±4cm)
TREADMILL_STARTUP_BIAS_V    = -0.006 # 起動時のオフセット打ち消し電圧

# 安全リミット
TREADMILL_MAX_VOLTAGE      = 2.5   # 最大出力電圧 [V]
TREADMILL_RAMP_V_PER_S     = 0.7   # 電圧変化率の上限 [V/s]
TREADMILL_ESTOP_DISTANCE_M = 1.0   # 緊急停止する距離 [m]


# ============================================================
#  剛体ID定義（Motive Streaming ID）
# ============================================================

HEAD_INDEX                  = 1
TRUNK_INDEX                 = 2
SACRAL_INDEX                = 3
R_ILIUM_INDEX               = 4
L_ILIUM_INDEX               = 5
R_GREATER_TROCHANTER_INDEX  = 6
L_GREATER_TROCHANTER_INDEX  = 7
R_KNEE_INDEX                = 8
L_KNEE_INDEX                = 9
R_MALLEOLUS_INDEX           = 10
L_MALLEOLUS_INDEX           = 11
R_TOE_INDEX                 = 12
L_TOE_INDEX                 = 13

# ============================================================
#  画面解像度・UI関連
# ============================================================

RAW_W, RAW_H = GetSystemMetrics(0), GetSystemMetrics(1)
SCREEN_W     = 3840 if (RAW_W, RAW_H) != (3840, 2160) else RAW_W
SCREEN_H     = 2160 if (RAW_W, RAW_H) != (3840, 2160) else RAW_H

# ============================================================
#  正規化係数（学習時データの min / max）
# ============================================================

MAX_ARR = np.array([
    31.985029151356574,
    29.080972711296017,
    16.82619515029058,
    24.607153393495167,
    16.83723757446406,
    25.778015477301224,
    77.52878513736748,
    20.119311584890347,
    77.52891604977866,
    33.37114415914728,
    46.50135930491923,
    242.1102110294689,
    46.499495940399676,
    251.85576196409394,
    2.4010760614202655,
    4.997153765415635,
    18.991778833671116,
    1.3926982521132978,
    4.295441859196978,
    18.179283925572577
], dtype=float)

MIN_ARR = np.array([
    -31.984858278863406,
    -29.081255132804614,
    -39.72926562180796,
    -29.2293139816393,
    -40.45881522780963,
    -30.01374288166156,
    -12.368392312733464,
    -31.885448066250365,
    -14.360798360594693,
    -20.119865313799078,
    -18.35325287980943,
    -252.04716977448723,
    -18.23490507179852,
    -241.60776899974223,
    -1.4679137224066803,
    -3.825092068538877,
    -0.8754912879916857,
    -2.4046151649424217,
    -3.9124625148431598,
    -0.7974464202168049
], dtype=float)


# ============================================================
#  UI 表示用ラベル・並び順
# ============================================================

LABELS = [
    "Normalized change\ntrunk angle yz [-]",
    "Normalized change\ntrunk angle zx [-]",
    "Normalized change\nR hip angle yz [-]",
    "Normalized change\nR hip angle zx [-]",
    "Normalized change\nL hip angle yz [-]",
    "Normalized change\nL hip angle zx [-]",
    "Normalized change\nR knee angle yz [-]",
    "Normalized change\nR knee angle zx [-]",
    "Normalized change\nL knee angle yz [-]",
    "Normalized change\nL knee angle zx [-]",
    "Normalized change\nR ankle angle yz [-]",
    "Normalized change\nR ankle angle zx [-]",
    "Normalized change\nL ankle angle yz [-]",
    "Normalized change\nL ankle angle zx [-]",
    "Normalized\nR ground reaction force x [-]",
    "Normalized\nR ground reaction force y [-]",
    "Normalized\nR ground reaction force z [-]",
    "Normalized\nL ground reaction force x [-]",
    "Normalized\nL ground reaction force y [-]",
    "Normalized\nL ground reaction force z [-]",
]

DISPLAY_ORDER = [
    0, 4, 8, 12, 16,
    1, 5, 9, 13, 17,
    2, 6, 10, 14, 18,
    3, 7, 11, 15, 19,
]
