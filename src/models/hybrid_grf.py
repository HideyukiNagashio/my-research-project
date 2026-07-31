import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
from .graph_utils import create_graph_structure

class HybridGRFModel(nn.Module):
    """
    2. GNN + Transformerのハイブリッドモデル
    """
    def __init__(self, input_dim=14, output_dim=3, use_shortcut=False, seq_len=200, 
                 gnn_out_dim=16, cnn_pool_dim=32, 
                 d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout_prob=0.1):
        super(HybridGRFModel, self).__init__()
        
        # d_model corresponds to the dimension input to the transformer.
        imu_dim = input_dim - 8
        combined_dim = cnn_pool_dim + imu_dim
        
        # グラフ構造の初期化 (register_bufferでGPUデバイス転送に対応)
        norm_coords, edge_index, edge_weight = create_graph_structure(use_shortcut)
        self.register_buffer('norm_coords', norm_coords)
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_weight', edge_weight)
        
        # --- 3. GNN層 ---
        # 入力ノード特徴量: [圧力値(1), x(1), y(1)] = 3次元
        self.conv1 = GCNConv(3, gnn_out_dim)
        self.conv2 = GCNConv(gnn_out_dim, gnn_out_dim)
        
        # --- 4. プーリング層 ---
        # 8ノード分の特徴をFlattenし、全結合層で圧縮
        self.fc_pool = nn.Linear(8 * gnn_out_dim, cnn_pool_dim)
        
        # 5. 特徴量の次元をTransformerのd_modelに揃えるための射影層
        self.fc_proj = nn.Linear(combined_dim, d_model)
        
        # --- 6. Transformer層 ---
        # 位置エンコーディング (時系列情報を付与)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        # Transformer (バッチファースト)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout_prob,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # --- 7. 出力層 ---
        # 3軸の床反力 (Fx, Fy, Fz)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        """
        入力: x の形状は (Batch, 200, 14)
        """
        batch_size, seq_len, num_features = x.size()
        
        # 1. データ分離
        pressure = x[:, :, :8] # (Batch, 200, 8)
        imu = x[:, :, 8:]      # (Batch, 200, 6)
        
        # 2. ノード特徴量作成
        # 圧力をノード特徴量にするため変形: (Batch * 200, 8, 1)
        p_reshaped = pressure.reshape(batch_size * seq_len, 8, 1)
        
        # 座標をバッチ＆時系列分に拡張: (Batch * 200, 8, 2)
        coords_expanded = self.norm_coords.unsqueeze(0).expand(batch_size * seq_len, 8, 2)
        
        # 結合してノード特徴量を作成: (Batch * 200, 8, 3)  ※[圧力, x, y]
        node_features = torch.cat([p_reshaped, coords_expanded], dim=2)
        
        # PyGに渡すためノード次元でFlatten: (Batch * 200 * 8, 3)
        node_features = node_features.reshape(batch_size * seq_len * 8, 3)
        
        # --- GNN用のバッチエッジ生成 ---
        # 全タイムステップ・全バッチのグラフを1つの大きな非連結グラフとして計算するためのオフセット処理
        num_graphs = batch_size * seq_len
        offset = torch.arange(0, num_graphs * 8, 8, device=x.device).view(-1, 1, 1)
        
        batched_edge_index = self.edge_index.unsqueeze(0).expand(num_graphs, 2, -1)
        batched_edge_index = (batched_edge_index + offset).transpose(0, 1).reshape(2, -1) # (2, num_graphs * E)
        batched_edge_weight = self.edge_weight.unsqueeze(0).expand(num_graphs, -1).reshape(-1)
        
        # 3. GNN層 (空間特徴抽出)
        g_out = F.relu(self.conv1(node_features, batched_edge_index, batched_edge_weight))
        g_out = F.relu(self.conv2(g_out, batched_edge_index, batched_edge_weight)) # (Batch*200*8, 16)
        
        # 4. プーリング
        # Flatten: (Batch * 200, 8 * 16)
        g_out = g_out.reshape(batch_size * seq_len, 8 * self.conv2.out_channels)
        
        # 圧縮: (Batch * 200, 32)
        pooled = F.relu(self.fc_pool(g_out))
        
        # 形状を元の系列に戻す: (Batch, 200, 32)
        pooled = pooled.reshape(batch_size, seq_len, -1)
        
        # 5. 結合
        # GNN特徴(cnn_pool_dim)とIMU特徴(6)を結合
        combined = torch.cat([pooled, imu], dim=2) 
        
        # 射影: combined_dim -> d_model
        combined_proj = self.fc_proj(combined)
        
        # 6. Transformer層 (時間特徴抽出)
        # 位置エンコーディングを加算
        combined_proj = combined_proj + self.pos_embedding
        transformer_out = self.transformer(combined_proj) # (Batch, 200, d_model)
        
        # 7. 出力層
        # (Batch, 200, 3) -> 時系列各ステップでの(Fx, Fy, Fz)
        out = self.fc_out(transformer_out) 
        
        return out
