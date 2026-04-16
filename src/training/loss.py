import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self, weights_list, device):
        """
        特定の出力変数の損失を強調するためのMSE。
        
        Args:
            weights_list (list): 出力特徴量と同次元の重みリスト
            device (torch.device): デバイス
        """
        super(WeightedMSELoss, self).__init__()
        self.weights = torch.tensor(weights_list, dtype=torch.float32).to(device)
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        loss = self.mse(pred, target)
        
        # broadcasting を利用して重み付け
        # pred と target は通常 (Batch, Time, Features) 
        # self.weights は (Features,) と想定
        weighted_loss = loss * self.weights
        
        return weighted_loss.mean()
