import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    複数サンプルの予測結果に対する評価指標を計算する。
    y_true, y_pred は (N, T, F) または flattened array でも動作するよう 2D/1D に変換して処理。
    """
    y_t = y_true.flatten()
    y_p = y_pred.flatten()

    mse = mean_squared_error(y_t, y_p)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_t, y_p)
    r2 = r2_score(y_t, y_p)
    
    # NRMSE (正規化RMSE) = RMSE / (max(y_true) - min(y_true))
    # 全データに基づく標準化
    ptp = np.ptp(y_t)
    nrmse = rmse / ptp if ptp != 0 else 0.0

    return {
        'rmse': rmse,
        'nrmse': nrmse,
        'mae': mae,
        'r2': r2
    }
