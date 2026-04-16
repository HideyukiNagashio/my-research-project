import torch
import torch.nn as nn

class TimeSeriesCNNRegression(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, output_dim, dropout_prob=0.3):
        super(TimeSeriesCNNRegression, self).__init__()
        
        # Padding='same' のために padding = kernel_size // 2 を設定
        pad = kernel_size // 2

        # --- 特徴抽出ブロック (Feature Extractor) ---
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=pad),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            # Block 2
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            # Block 3
            nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        
        # --- 回帰ヘッド (Regressor Head) ---
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        # x shape: (Batch, Time, Features) -> (Batch, Features, Time)
        x = x.permute(0, 2, 1)
        
        x = self.features(x)
        
        # (Batch, Features, Time) -> (Batch, Time, Features)
        x = x.permute(0, 2, 1)
        
        output = self.regressor(x)
        return output
