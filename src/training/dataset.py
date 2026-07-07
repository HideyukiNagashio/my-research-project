import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from src.preprocessing.feature_selector import FeatureSelector

class GaitDataset(Dataset):
    def __init__(self, pkl_path: str, input_type: str = 'bilateral', target_type: str = 'all',
                 stride_type_X: str = '1.0_post_swing', stride_type_Y: str = '1.0_post_swing'):
        """
        Cross Validation の Fold別PickleファイルからDatasetを構成する。
        
        Args:
            pkl_path (str): 'train.pkl', 'val.pkl', または 'test.pkl' のパス
            input_type (str): 'single_leg', 'bilateral', 'pressure_single', 'pressure_bilateral', 'imu_single', 'imu_bilateral'
            target_type (str): 'all', 'angles_only', 'grf_only'
            stride_type_X (str): 入力(X)の周期タイプ ('0.5', '1.0', '1.5')
            stride_type_Y (str): 出力(Y)の周期タイプ ('0.5', '1.0')
        """
        super().__init__()
        
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            
        key_X = f'ensemble_X_{stride_type_X}'
        key_Y = f'ensemble_Y_{stride_type_Y}'
        
        if key_X in data and key_Y in data:
            ensemble_X = data[key_X]
            ensemble_Y = data[key_Y]
        else:
            # 古いデータ形式に対する後方互換性
            ensemble_X = data.get('ensemble', data.get(f'ensemble_X_{stride_type_X}'))
            ensemble_Y = data.get('ensemble', data.get(f'ensemble_Y_{stride_type_Y}'))
            if ensemble_X is None or ensemble_Y is None:
                raise KeyError(f"Expected keys '{key_X}' and '{key_Y}', or 'ensemble' not found in {pkl_path}")
        
        # Target (y) の切り出し
        if target_type == 'all':
            self.y = FeatureSelector.get_y(ensemble_Y)
        elif target_type == 'angles_only':
            self.y = FeatureSelector.get_y_angles_only(ensemble_Y)
        elif target_type == 'angles_6dof':
            self.y = FeatureSelector.get_y_angles_6dof(ensemble_Y)
        elif target_type == 'angles_3dof':
            self.y = FeatureSelector.get_y_angles_3dof(ensemble_Y)
        elif target_type == 'grf_only':
            self.y = FeatureSelector.get_y_grf_only(ensemble_Y)
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
            
        # Input (X) の切り出し
        if input_type == 'single_leg':
            self.X = FeatureSelector.get_X_single_leg(ensemble_X)
        elif input_type == 'bilateral':
            self.X = FeatureSelector.get_X_bilateral(ensemble_X)
        elif input_type == 'pressure_single':
            self.X = FeatureSelector.get_X_pressure_only(ensemble_X, bilateral=False)
        elif input_type == 'pressure_bilateral':
            self.X = FeatureSelector.get_X_pressure_only(ensemble_X, bilateral=True)
        elif input_type == 'imu_single':
            self.X = FeatureSelector.get_X_imu_only(ensemble_X, bilateral=False)
        elif input_type == 'imu_bilateral':
            self.X = FeatureSelector.get_X_imu_only(ensemble_X, bilateral=True)
        else:
            raise ValueError(f"Unknown input_type: {input_type}")
            
        # Pytorch用Tensorに変換 (dtype=float32)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)
        
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
