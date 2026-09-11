import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from .graph_utils import create_graph_structure
from .transformer import PositionalEncoding

class HybridGRFNoCoordsModel(nn.Module):
    """
    GCNのノード特徴量から座標情報を削除し、足底圧力（1ch）のみを入力とするハイブリッドモデル。
    グラフの構造（edge_index, edge_weight）には引き続き座標由来の距離情報が使用される。
    """
    def __init__(self, input_dim=14, output_dim=3, use_shortcut=False, seq_len=200, 
                 gnn_out_dim=16, cnn_pool_dim=32, 
                 d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout_prob=0.1):
        super(HybridGRFNoCoordsModel, self).__init__()
        self.d_model = d_model
        
        imu_dim = input_dim - 8
        combined_dim = cnn_pool_dim + imu_dim
        
        # グラフ構造の初期化
        # 座標(norm_coords)はここではノード特徴量としては使わないが、
        # edge_weightには既にgraph_utils内で「1/ユークリッド距離」として反映されている。
        norm_coords, edge_index, edge_weight = create_graph_structure(use_shortcut)
        self.register_buffer('norm_coords', norm_coords)
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_weight', edge_weight)
        
        # --- GNN層 ---
        # 入力ノード特徴量: [圧力値]のみ = 1次元
        self.conv1 = GCNConv(1, gnn_out_dim)
        self.conv2 = GCNConv(gnn_out_dim, gnn_out_dim)
        
        # --- プーリング層 ---
        self.fc_pool = nn.Linear(8 * gnn_out_dim, cnn_pool_dim)
        
        # --- 射影層 ---
        self.fc_proj = nn.Linear(combined_dim, d_model)
        
        # --- Transformer層 ---
        self.pos_encoder = PositionalEncoding(self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout_prob,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # --- 出力層 ---
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        batch_size, seq_len, num_features = x.size()
        
        # 1. データ分離
        pressure = x[:, :, :8] # (B, 200, 8)
        imu = x[:, :, 8:]      # (B, 200, 6)
        
        # 2. ノード特徴量作成
        # 圧力のみをノード特徴量とする: (B*200, 8, 1)
        node_features = pressure.reshape(batch_size * seq_len, 8, 1)
        
        # PyGのバッチ処理用にFlatten: (B*200*8, 1)
        node_features = node_features.reshape(batch_size * seq_len * 8, 1)
        
        # --- GNN用のバッチエッジ生成 ---
        num_graphs = batch_size * seq_len
        offset = torch.arange(0, num_graphs * 8, 8, device=x.device).view(-1, 1, 1)
        
        batched_edge_index = self.edge_index.unsqueeze(0).expand(num_graphs, 2, -1)
        batched_edge_index = (batched_edge_index + offset).transpose(0, 1).reshape(2, -1)
        batched_edge_weight = self.edge_weight.unsqueeze(0).expand(num_graphs, -1).reshape(-1)
        
        # 3. GNN層 (ノード特徴量と、座標由来のedge_weightを使用)
        g_out = F.relu(self.conv1(node_features, batched_edge_index, batched_edge_weight)) # (B*200*8, 16)
        g_out = F.relu(self.conv2(g_out, batched_edge_index, batched_edge_weight))         # (B*200*8, 16)
        
        # 4. プーリング
        g_out = g_out.reshape(batch_size * seq_len, 8 * self.conv2.out_channels) # (B*200, 128)
        pooled = F.relu(self.fc_pool(g_out))                                     # (B*200, 32)
        pooled = pooled.reshape(batch_size, seq_len, -1)                         # (B, 200, 32)
        
        # 5. 結合と射影
        combined = torch.cat([pooled, imu], dim=2) # (B, 200, 38)
        combined_proj = self.fc_proj(combined)     # (B, 200, 128)
        
        # 6. Transformer層
        combined_proj = combined_proj * math.sqrt(self.d_model)
        combined_proj = self.pos_encoder(combined_proj)
        transformer_out = self.transformer(combined_proj) # (B, 200, 128)
        
        # 7. 出力層
        out = self.fc_out(transformer_out) # (B, 200, 3)
        
        return out
