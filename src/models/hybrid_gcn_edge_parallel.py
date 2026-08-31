import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, EdgeConv
from .graph_utils import create_graph_structure

class HybridGCNEdgeParallelModel(nn.Module):
    def __init__(self, input_dim=14, output_dim=3, use_shortcut=False, seq_len=200, 
                 gnn_out_dim=16, cnn_pool_dim=32, 
                 d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout_prob=0.1):
        super(HybridGCNEdgeParallelModel, self).__init__()
        
        self.gnn_out_dim = gnn_out_dim
        imu_dim = input_dim - 8
        combined_dim = cnn_pool_dim + imu_dim
        
        # グラフ構造の初期化 (GCNはエッジ重みを使用、EdgeConvはエッジインデックスのみ使用)
        norm_coords, edge_index, edge_weight = create_graph_structure(use_shortcut)
        self.register_buffer('norm_coords', norm_coords)
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_weight', edge_weight)
        
        # --- Branch A: GCN ---
        self.gcn_conv1 = GCNConv(3, gnn_out_dim)
        self.gcn_conv2 = GCNConv(gnn_out_dim, gnn_out_dim)
        
        # --- Branch B: EdgeConv ---
        # EdgeConvは 2 * in_channels (自身のノード特徴量 + 隣接ノード特徴量との差分) を受け取るMLPを定義する
        mlp1 = nn.Sequential(
            nn.Linear(2 * 3, gnn_out_dim),
            nn.ReLU(),
            nn.Linear(gnn_out_dim, gnn_out_dim)
        )
        self.edge_conv1 = EdgeConv(nn=mlp1, aggr='mean')
        
        mlp2 = nn.Sequential(
            nn.Linear(2 * gnn_out_dim, gnn_out_dim),
            nn.ReLU(),
            nn.Linear(gnn_out_dim, gnn_out_dim)
        )
        self.edge_conv2 = EdgeConv(nn=mlp2, aggr='mean')
        
        # --- プーリング層 ---
        # 2つのブランチの出力をConcatするため、ノードあたりの特徴量次元は 2 * gnn_out_dim になる
        # それを8ノード分Flattenしてプーリング
        self.fc_pool = nn.Linear(8 * (2 * gnn_out_dim), cnn_pool_dim)
        
        # --- Transformer統合 ---
        self.fc_proj = nn.Linear(combined_dim, d_model)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout_prob, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # --- 出力層 ---
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # 1. データ分離
        pressure = x[:, :, :8]
        imu = x[:, :, 8:]
        
        # 2. ノード特徴量作成: [圧力, x, y]
        p_reshaped = pressure.reshape(batch_size * seq_len, 8, 1)
        coords_expanded = self.norm_coords.unsqueeze(0).expand(batch_size * seq_len, 8, 2)
        node_features = torch.cat([p_reshaped, coords_expanded], dim=2)
        node_features = node_features.reshape(batch_size * seq_len * 8, 3)
        
        # --- バッチエッジ生成 ---
        num_graphs = batch_size * seq_len
        offset = torch.arange(0, num_graphs * 8, 8, device=x.device).view(-1, 1, 1)
        
        batched_edge_index = self.edge_index.unsqueeze(0).expand(num_graphs, 2, -1)
        batched_edge_index = (batched_edge_index + offset).transpose(0, 1).reshape(2, -1)
        batched_edge_weight = self.edge_weight.unsqueeze(0).expand(num_graphs, -1).reshape(-1)
        
        # --- 3. Branch A: GCN処理 ---
        gcn_out = F.relu(self.gcn_conv1(node_features, batched_edge_index, batched_edge_weight))
        gcn_out = F.relu(self.gcn_conv2(gcn_out, batched_edge_index, batched_edge_weight))
        
        # --- 4. Branch B: EdgeConv処理 ---
        # 既存の静的な edge_index を用いてEdgeConvを適用する
        edge_out = F.relu(self.edge_conv1(node_features, batched_edge_index))
        edge_out = F.relu(self.edge_conv2(edge_out, batched_edge_index))
        
        # --- 5. 特徴量結合とプーリング ---
        # GCN出力とEdgeConv出力を特徴次元(dim=1)で結合: shape (B*200*8, 32)
        combined_gout = torch.cat([gcn_out, edge_out], dim=1)
        
        # 8ノード分をFlatten: shape (B*200, 8 * 32)
        combined_gout = combined_gout.reshape(batch_size * seq_len, 8 * (2 * self.gnn_out_dim))
        
        # プーリング層: (B*200, 32) -> (B, 200, 32)
        pooled = F.relu(self.fc_pool(combined_gout))
        pooled = pooled.reshape(batch_size, seq_len, -1)
        
        # --- 6. IMUとの結合とTransformer処理 ---
        combined = torch.cat([pooled, imu], dim=2)
        combined_proj = self.fc_proj(combined)
        
        combined_proj = combined_proj + self.pos_embedding
        transformer_out = self.transformer(combined_proj)
        out = self.fc_out(transformer_out)
        
        return out
