from .cnn import TimeSeriesCNNRegression
from .bilstm import AdvancedBiLSTMRegression
from .transformer import TimeSeriesTransformer
from .transformer_GeLU import TimeSeriesTransformer as TimeSeriesTransformerGeLU
from .hybrid_grf import HybridGRFModel

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
    elif model_name == 'transformer_gelu':
        return TimeSeriesTransformerGeLU(**kwargs)
    elif model_name == 'hybrid_grf':
        return HybridGRFModel(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Supported models are: cnn, bilstm, transformer, transformer_gelu, hybrid_grf.")

