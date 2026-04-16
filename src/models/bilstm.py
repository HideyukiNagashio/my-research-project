import torch
import torch.nn as nn

class AdvancedBiLSTMRegression(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob=0.3):
        super(AdvancedBiLSTMRegression, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout_prob if num_layers > 1 else 0,
            bidirectional=True 
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        r_out, _ = self.lstm(x)
        r_out = self.layer_norm(r_out)
        output = self.fc(r_out)
        return output
