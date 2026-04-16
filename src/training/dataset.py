import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from src.preprocessing.feature_selector import FeatureSelector

class GaitDataset(Dataset):
    def __init__(self, pkl_path: str, input_type: str = 'bilateral', target_type: str = 'all'):
        """
        Cross Validation の Fold別PickleファイルからDatasetを構成する。
        
        Args:
            pkl_path (str): 'train.pkl', 'val.pkl', または 'test.pkl' のパス
            input_type (str): 'single_leg', 'bilateral', 'pressure_single', 'pressure_bilateral', 'imu_single', 'imu_bilateral'
            target_type (str): 'all', 'angles_only', 'grf_only'
        """
        super().__init__()
        
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            
        ensemble = data['ensemble'] # (N, T, 52)
        
        # Target (y) の切り出し
        if target_type == 'all':
            self.y = FeatureSelector.get_y(ensemble)
        elif target_type == 'angles_only':
            self.y = FeatureSelector.get_y_angles_only(ensemble)
        elif target_type == 'grf_only':
            self.y = FeatureSelector.get_y_grf_only(ensemble)
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
            
        # Input (X) の切り出し
        if input_type == 'single_leg':
            self.X = FeatureSelector.get_X_single_leg(ensemble)
        elif input_type == 'bilateral':
            self.X = FeatureSelector.get_X_bilateral(ensemble)
        elif input_type == 'pressure_single':
            self.X = FeatureSelector.get_X_pressure_only(ensemble, bilateral=False)
        elif input_type == 'pressure_bilateral':
            self.X = FeatureSelector.get_X_pressure_only(ensemble, bilateral=True)
        elif input_type == 'imu_single':
            self.X = FeatureSelector.get_X_imu_only(ensemble, bilateral=False)
        elif input_type == 'imu_bilateral':
            self.X = FeatureSelector.get_X_imu_only(ensemble, bilateral=True)
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
