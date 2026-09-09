import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import EdgeConv
from .graph_utils import create_graph_structure
from .transformer import PositionalEncoding

class HybridEdgeConvModelC(nn.Module):
    """
    EdgeConvハイブリッドモデル (モデルC)
    - 位置エンコーディング: Sin/Cos (固定)
    - スケール調整: あり (* sqrt(d_model))
    - 最終出力層: 1層 (Linear)
    """
    def __init__(self, input_dim=14, output_dim=3, use_shortcut=False, seq_len=200, 
                 gnn_out_dim=16, cnn_pool_dim=32, 
                 d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout_prob=0.1, **kwargs):
        super(HybridEdgeConvModelC, self).__init__()
        
        self.d_model = d_model
        self.gnn_out_dim = gnn_out_dim
        imu_dim = input_dim - 8
        combined_dim = cnn_pool_dim + imu_dim
        
        norm_coords, edge_index, _ = create_graph_structure(use_shortcut)
        self.register_buffer('norm_coords', norm_coords)
        self.register_buffer('edge_index', edge_index)
        
        mlp1 = nn.Sequential(
            nn.Linear(2 * 3, gnn_out_dim),
            nn.ReLU(),
            nn.Linear(gnn_out_dim, gnn_out_dim)
        )
        self.conv1 = EdgeConv(nn=mlp1, aggr='mean')
        
        mlp2 = nn.Sequential(
            nn.Linear(2 * gnn_out_dim, gnn_out_dim),
            nn.ReLU(),
            nn.Linear(gnn_out_dim, gnn_out_dim)
        )
        self.conv2 = EdgeConv(nn=mlp2, aggr='mean')
        
        self.fc_pool = nn.Linear(8 * gnn_out_dim, cnn_pool_dim)
        self.fc_proj = nn.Linear(combined_dim, d_model)
        
        # モデルCの仕様1: 固定のSin/Cos位置エンコーディング
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout_prob, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # モデルCの仕様2: 最終出力層は1層
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        pressure = x[:, :, :8]
        imu = x[:, :, 8:]
        
        p_reshaped = pressure.reshape(batch_size * seq_len, 8, 1)
        coords_expanded = self.norm_coords.unsqueeze(0).expand(batch_size * seq_len, 8, 2)
        node_features = torch.cat([p_reshaped, coords_expanded], dim=2)
        node_features = node_features.reshape(batch_size * seq_len * 8, 3) 
        
        num_graphs = batch_size * seq_len
        offset = torch.arange(0, num_graphs * 8, 8, device=x.device).view(-1, 1, 1)
        batched_edge_index = self.edge_index.unsqueeze(0).expand(num_graphs, 2, -1)
        batched_edge_index = (batched_edge_index + offset).transpose(0, 1).reshape(2, -1)
        
        g_out = F.relu(self.conv1(node_features, batched_edge_index)) 
        g_out = F.relu(self.conv2(g_out, batched_edge_index)) 
        
        g_out = g_out.reshape(batch_size * seq_len, 8 * self.gnn_out_dim) 
        pooled = F.relu(self.fc_pool(g_out)) 
        pooled = pooled.reshape(batch_size, seq_len, -1) 
        
        combined = torch.cat([pooled, imu], dim=2) 
        combined_proj = self.fc_proj(combined) 
        
        # モデルCの仕様3: スケール調整あり (* sqrt(d_model))
        combined_proj = combined_proj * math.sqrt(self.d_model)
        combined_proj = self.pos_encoder(combined_proj)
        
        transformer_out = self.transformer(combined_proj)
        out = self.fc_out(transformer_out)
        return out
