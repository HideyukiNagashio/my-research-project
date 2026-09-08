import torch
import torch.nn as nn
import math

class AlignedTimeSeriesTransformer(nn.Module):
    """
    ハイブリッドモデル（GCN/EdgeConv）と位置エンコーディングや最終出力層の
    仕様を完全に揃えたノーマルTransformerモデル。
    空間特徴抽出（GNN等）の有無による純粋な比較（アブレーションスタディ）用。
    """
    def __init__(self, input_dim=14, output_dim=3, seq_len=200, 
                 d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout_prob=0.1, **kwargs):
        super(AlignedTimeSeriesTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim
        
        # 1. 入力特徴量の射影 (14 -> 128)
        self.fc_proj = nn.Linear(self.input_dim, self.d_model)
        
        # 2. 学習可能な位置エンコーディング (ハイブリッドモデルと統一)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, self.d_model))
        
        # 3. Transformer Encoder (バッチファースト)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout_prob,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. 最終出力層 (ハイブリッドモデルと統一：1層のLinear)
        self.fc_out = nn.Linear(self.d_model, self.output_dim)

    def forward(self, x):
        """
        入力: x の形状は (Batch, seq_len, input_dim)
        """
        # 特徴量の射影
        x = self.fc_proj(x) # shape: (B, seq_len, d_model)
        
        # 学習可能な位置エンコーディングを加算（スケール調整なし）
        x = x + self.pos_embedding
        
        # Transformer層 (時間特徴抽出)
        r_out = self.transformer_encoder(x) # shape: (B, seq_len, d_model)
        
        # 最終出力層
        output = self.fc_out(r_out) # shape: (B, seq_len, output_dim)
        
        return output
