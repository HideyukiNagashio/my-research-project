import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def _calc_scores(y_t, y_p):
    mse = mean_squared_error(y_t, y_p)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_t, y_p)
    
    # R2 score は分散0の場合にエラーを防ぐ
    if np.var(y_t) == 0:
        r2 = 0.0
    else:
        r2 = r2_score(y_t, y_p)
        
    ptp = np.ptp(y_t)
    nrmse = rmse / ptp if ptp != 0 else 0.0
    
    return {
        'rmse': rmse,
        'nrmse': nrmse,
        'mae': mae,
        'r2': r2
    }

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    複数サンプルの予測結果に対する評価指標を、全体(overall)と特徴量別(per_feature)に計算する。
    """
    # 全体の指標計算
    y_t_flat = y_true.flatten()
    y_p_flat = y_pred.flatten()
    overall = _calc_scores(y_t_flat, y_p_flat)
    
    # 特徴量別の指標計算
    # y_true, y_pred は通常 (N, T, F)。これを (N*T, F) にリシェイプして計算
    if y_true.ndim == 3:
        y_true_2d = y_true.reshape(-1, y_true.shape[-1])
        y_pred_2d = y_pred.reshape(-1, y_pred.shape[-1])
    elif y_true.ndim == 2:
        y_true_2d = y_true
        y_pred_2d = y_pred
    else:
        # flattenedの場合特徴量別は出さないがキーは用意する
        overall['per_feature'] = []
        return overall

    per_feature = []
    for f_idx in range(y_true_2d.shape[-1]):
        feat_scores = _calc_scores(y_true_2d[:, f_idx], y_pred_2d[:, f_idx])
        per_feature.append(feat_scores)
        
    overall['per_feature'] = per_feature
    return overall
