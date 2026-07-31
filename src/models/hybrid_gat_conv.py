import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from .graph_utils import create_graph_structure

class HybridGATConvModel(nn.Module):
    def __init__(self, input_dim=14, output_dim=3, use_shortcut=False, seq_len=200, 
                 gnn_out_dim=16, cnn_pool_dim=32, 
                 d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout_prob=0.1):
        super(HybridGATConvModel, self).__init__()
        
        self.gnn_out_dim = gnn_out_dim
        imu_dim = input_dim - 8
        combined_dim = cnn_pool_dim + imu_dim
        
        # GAT: Use fully connected graph
        norm_coords, edge_index, _ = create_graph_structure(use_shortcut=False, fully_connected=True)
        self.register_buffer('norm_coords', norm_coords)
        self.register_buffer('edge_index', edge_index)
        
        gat_heads = 2
        gat_out_channels = gnn_out_dim // gat_heads # 16 // 2 = 8
        
        # GATConv 1 (out = 8 * 2 = 16)
        self.conv1 = GATConv(3, gat_out_channels, heads=gat_heads, concat=True)
        # GATConv 2 (out = 8 * 2 = 16)
        self.conv2 = GATConv(gnn_out_dim, gat_out_channels, heads=gat_heads, concat=True)
        
        self.fc_pool = nn.Linear(8 * gnn_out_dim, cnn_pool_dim)
        self.fc_proj = nn.Linear(combined_dim, d_model)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout_prob, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        pressure = x[:, :, :8]
        imu = x[:, :, 8:]
        
        p_reshaped = pressure.reshape(batch_size * seq_len, 8, 1)
        coords_expanded = self.norm_coords.unsqueeze(0).expand(batch_size * seq_len, 8, 2)
        node_features = torch.cat([p_reshaped, coords_expanded], dim=2)
        node_features = node_features.reshape(batch_size * seq_len * 8, 3) # shape: (B*200*8, 3)
        
        num_graphs = batch_size * seq_len
        offset = torch.arange(0, num_graphs * 8, 8, device=x.device).view(-1, 1, 1)
        batched_edge_index = self.edge_index.unsqueeze(0).expand(num_graphs, 2, -1)
        batched_edge_index = (batched_edge_index + offset).transpose(0, 1).reshape(2, -1)
        
        g_out = F.relu(self.conv1(node_features, batched_edge_index)) # shape: (B*200*8, 16)
        g_out = F.relu(self.conv2(g_out, batched_edge_index)) # shape: (B*200*8, 16)
        
        g_out = g_out.reshape(batch_size * seq_len, 8 * self.gnn_out_dim) # shape: (B*200, 128)
        pooled = F.relu(self.fc_pool(g_out)) # shape: (B*200, 32)
        pooled = pooled.reshape(batch_size, seq_len, -1) # shape: (B, 200, 32)
        
        combined = torch.cat([pooled, imu], dim=2) # shape: (B, 200, 38)
        combined_proj = self.fc_proj(combined) # shape: (B, 200, d_model)
        
        combined_proj = combined_proj + self.pos_embedding
        transformer_out = self.transformer(combined_proj)
        out = self.fc_out(transformer_out)
        return out
