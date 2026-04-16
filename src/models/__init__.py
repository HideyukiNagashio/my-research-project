from .cnn import TimeSeriesCNNRegression
from .bilstm import AdvancedBiLSTMRegression
from .transformer import TimeSeriesTransformer

def get_model(model_name: str, **kwargs):
    """
    モデル名とハイパーパラメータからモデルインスタンスを生成するファクトリ関数
    """
    model_name = model_name.lower()
    if model_name == 'cnn':
        return TimeSeriesCNNRegression(**kwargs)
    elif model_name == 'bilstm':
        return AdvancedBiLSTMRegression(**kwargs)
    elif model_name == 'transformer':
        return TimeSeriesTransformer(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Supported models are: cnn, bilstm, transformer.")
