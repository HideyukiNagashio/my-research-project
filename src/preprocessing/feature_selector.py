import numpy as np

class FeatureSelector:
    """
    歩行データのアンサンブルテンソル (N, T, 52) から、
    目的のモデル入力 (X)とターゲット (y)を任意に切り出すためのユーティリティ。

    【52列の特徴量インデックス内訳】
    ■ 同側 (Ipsilateral): 0〜25
      [0 - 7]   : Pressure 1-8 (足底圧力)
      [8 - 10]  : Accel X, Y, Z (加速度)
      [11 - 13] : Gyro X, Y, Z (ジャイロ)
      [14 - 16] : Hip Angles (Flex/Ext, Abd/Add, Int/Ext)
      [17 - 19] : Knee Angles (Flex/Ext, Abd/Add, Int/Ext)
      [20 - 22] : Ankle Angles(Flex/Ext, Abd/Add, Int/Ext)
      [23 - 25] : GRF (Fx, Fy, Fz - 床反力)

    ■ 対側 (Contralateral): 26〜51
      [26 - 33] : Contra_Pressure 1-8
      [34 - 36] : Contra_Accel X, Y, Z
      [37 - 39] : Contra_Gyro X, Y, Z
      [40 - 42] : Contra_Hip Angles
      [43 - 45] : Contra_Knee Angles
      [46 - 48] : Contra_Ankle Angles
      [49 - 51] : Contra_GRF (Fx, Fy, Fz)
    """
    
    # ---------------------------------------------------------
    # ターゲット (y) 用抽出
    # ---------------------------------------------------------
    @staticmethod
    def get_y(ensemble):
        """全ターゲット: 同側の関節角度(9ch) + 床反力(3ch) = (N, T, 12)"""
        return ensemble[:, :, 14:26]

    @staticmethod
    def get_y_angles_only(ensemble):
        """ターゲット: 関節角度の予測のみ (Hip, Knee, Ankle の計9ch) = (N, T, 9)"""
        return ensemble[:, :, 14:23]

    @staticmethod
    def get_y_grf_only(ensemble):
        """ターゲット: 床反力の予測のみ (Fx, Fy, Fz の計3ch) = (N, T, 3)"""
        return ensemble[:, :, 23:26]

    # ---------------------------------------------------------
    # 入力 (X) 用抽出
    # ---------------------------------------------------------
    @staticmethod
    def _slice_features(ensemble, ipsi_idx, contra_idx=None):
        if contra_idx is not None:
            return np.concatenate([ensemble[:, :, ipsi_idx], ensemble[:, :, contra_idx]], axis=-1)
        else:
            return ensemble[:, :, ipsi_idx]

    @classmethod
    def get_X_single_leg(cls, ensemble):
        """片脚入力: 同側の Pressure(8ch) + IMU(6ch) = 14ch"""
        return cls._slice_features(ensemble, list(range(14)))

    @classmethod
    def get_X_bilateral(cls, ensemble):
        """両脚入力: 同側(14ch) + 対側(14ch) = 28ch"""
        return cls._slice_features(ensemble, list(range(14)), list(range(26, 40)))

    @classmethod
    def get_X_pressure_only(cls, ensemble, bilateral=True):
        """足底圧力のみ入力: 両足なら16ch、片足なら8ch"""
        ipsi_idx = list(range(8))
        contra_idx = list(range(26, 34)) if bilateral else None
        return cls._slice_features(ensemble, ipsi_idx, contra_idx)

    @classmethod
    def get_X_imu_only(cls, ensemble, bilateral=True):
        """IMUのみ入力: 両足なら12ch、片足なら6ch"""
        ipsi_idx = list(range(8, 14))
        contra_idx = list(range(34, 40)) if bilateral else None
        return cls._slice_features(ensemble, ipsi_idx, contra_idx)

