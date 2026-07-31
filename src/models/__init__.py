from .cnn import TimeSeriesCNNRegression
from .bilstm import AdvancedBiLSTMRegression
from .transformer import TimeSeriesTransformer
from .transformer_GeLU import TimeSeriesTransformer as TimeSeriesTransformerGeLU
from .hybrid_grf import HybridGRFModel
from .hybrid_edge_conv import HybridEdgeConvModel
from .hybrid_gat_conv import HybridGATConvModel
from .hybrid_gcn_residual import HybridGCNResidualModel
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
    elif model_name == 'hybrid_edge':
        return HybridEdgeConvModel(**kwargs)
    elif model_name == 'hybrid_gat':
        return HybridGATConvModel(**kwargs)
    elif model_name == 'hybrid_gcn_res':
        return HybridGCNResidualModel(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Supported models are: cnn, bilstm, transformer, transformer_gelu, hybrid_grf, hybrid_edge, hybrid_gat, hybrid_gcn_res.")
