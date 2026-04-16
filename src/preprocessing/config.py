import numpy as np

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
