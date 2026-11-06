import pickle
from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing.feature_selector import FeatureSelector

N_STANCE = 200
CONDITIONS = ('HS-TO', 'TO-TO', 'HS-HS')


@dataclass(frozen=True)
class WindowSpec:
    """入力系列内での立脚期の位置と，Positional Encoding の位置インデックスの定義．"""
    condition: str
    seq_len: int        # 入力系列長 T
    stance_start: int   # 立脚期の開始 index（入力系列内）
    stance_len: int     # 立脚期の点数
    pos_start: int      # 系列先頭の位置インデックス（i 番目の点は pos_start + i）
    n_swing: int        # 遊脚期の点数（HS-TO では 0）

    @property
    def stance_end(self):
        return self.stance_start + self.stance_len

    def to_dict(self):
        return asdict(self)


def get_window_spec(condition: str, n_swing: int = 140, n_stance: int = N_STANCE,
                    pe_mode: str = 'aligned') -> WindowSpec:
    """
    条件ごとの入力系列構成．
      HS-TO : [stance(200)]                    stance = [0, 200)        位置 0..199
      TO-TO : [swing_pre(n_swing), stance(200)] stance = [n_swing, T)    位置 -n_swing..-1, 0..199
      HS-HS : [stance(200), swing_post(n_swing)] stance = [0, 200)       位置 0..199, 200..199+n_swing

    pe_mode:
      'aligned'    : 立脚期の位置インデックスを全条件で 0..199 に揃える（既定）
      'sequential' : 系列先頭を常に 0 とする従来方式（TO-TO では立脚期が n_swing..T-1 になる）
    """
    if condition == 'HS-TO':
        spec = dict(seq_len=n_stance, stance_start=0, n_swing=0)
    elif condition == 'TO-TO':
        spec = dict(seq_len=n_swing + n_stance, stance_start=n_swing, n_swing=n_swing)
    elif condition == 'HS-HS':
        spec = dict(seq_len=n_stance + n_swing, stance_start=0, n_swing=n_swing)
    else:
        raise ValueError(f"Unknown condition: {condition} (expected one of {CONDITIONS})")

    if pe_mode == 'aligned':
        pos_start = -spec['stance_start']
    elif pe_mode == 'sequential':
        pos_start = 0
    else:
        raise ValueError(f"Unknown pe_mode: {pe_mode}")
    return WindowSpec(condition=condition, stance_len=n_stance, pos_start=pos_start, **spec)


def build_condition_ensemble(data: dict, condition: str):
    """
    pkl の辞書から，条件に対応する 52ch アンサンブル (N, T, 52) と立脚期の位置を返す．
    立脚期 200 点は常に ensemble_X_0.5（HS→TO を 200 点に正規化したもの = 既存の基準入力）を
    そのまま用いるため，立脚期の入力・教師は全条件で完全に同一になる．
    """
    stance = data['ensemble_X_0.5']
    if condition == 'HS-TO':
        return stance
    if condition == 'TO-TO':
        return np.concatenate([data['ensemble_S_pre_swing'], stance], axis=1)
    if condition == 'HS-HS':
        return np.concatenate([stance, data['ensemble_S_post_swing']], axis=1)
    raise ValueError(f"Unknown condition: {condition}")


class GaitSwingStanceDataset(Dataset):
    """
    遊脚期を含む入力 (N, T, 14) と，立脚期の床反力 (N, 200, 3) を返す Dataset．

    GaitDataset（既存）とは独立．既存の FeatureSelector をそのまま使うため，
    チャネル選択・前処理（鏡映・正規化）は既存と同一．
    """
    def __init__(self, pkl_path: str, condition: str, input_type: str = 'single_leg',
                 target_type: str = 'grf_only', n_swing: int = None, pe_mode: str = 'aligned'):
        super().__init__()
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        if n_swing is None:
            n_swing = data['ensemble_S_pre_swing'].shape[1] if 'ensemble_S_pre_swing' in data else 0
        self.spec = get_window_spec(condition, n_swing=n_swing, pe_mode=pe_mode)

        ens_X = build_condition_ensemble(data, condition)
        ens_Y = data['ensemble_Y_0.5']
        assert ens_X.shape[1] == self.spec.seq_len, (ens_X.shape, self.spec)
        assert ens_Y.shape[1] == self.spec.stance_len, ens_Y.shape

        if input_type == 'single_leg':
            X = FeatureSelector.get_X_single_leg(ens_X)
        elif input_type == 'bilateral':
            X = FeatureSelector.get_X_bilateral(ens_X)
        else:
            raise ValueError(f"input_type {input_type} is not supported for the swing/stance dataset")

        if target_type == 'grf_only':
            y = FeatureSelector.get_y_grf_only(ens_Y)
        else:
            raise ValueError(f"target_type {target_type} is not supported for the swing/stance dataset")

        # --- 入力の立脚期区間と教師データの時間的対応を保証する ---
        # Y_0.5 と同じ区間の入力（Y の 52ch のうち入力チャネル）が，入力系列の立脚期区間と完全一致すること
        Y_in = FeatureSelector.get_X_single_leg(ens_Y) if input_type == 'single_leg' \
            else FeatureSelector.get_X_bilateral(ens_Y)
        assert np.array_equal(X[:, self.spec.stance_start:self.spec.stance_end, :], Y_in), \
            "Stance window of the input does not match the target stride (Y_0.5)"

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        assert self.y.shape[1] == self.spec.stance_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @property
    def input_dim(self):
        return self.X.shape[-1]

    @property
    def output_dim(self):
        return self.y.shape[-1]

