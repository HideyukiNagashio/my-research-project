from .cnn import TimeSeriesCNNRegression
from .bilstm import AdvancedBiLSTMRegression
from .transformer import TimeSeriesTransformer
from .transformer_GeLU import TimeSeriesTransformer as TimeSeriesTransformerGeLU
from .hybrid_grf import HybridGRFModel
from .hybrid_edge_conv import HybridEdgeConvModel
from .hybrid_gat_conv import HybridGATConvModel
from .hybrid_gcn_residual import HybridGCNResidualModel
from .hybrid_gcn_edge_parallel import HybridGCNEdgeParallelModel
from .transformer_aligned import AlignedTimeSeriesTransformer
from .hybrid_grf_aligned import HybridGRFAlignedModel
from .hybrid_edge_aligned import HybridEdgeConvAlignedModel

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
    elif model_name == 'hybrid_gcn_edge_parallel':
        return HybridGCNEdgeParallelModel(**kwargs)
    elif model_name == 'transformer_aligned':
        return AlignedTimeSeriesTransformer(**kwargs)
    elif model_name == 'hybrid_grf_aligned':
        return HybridGRFAlignedModel(**kwargs)
    elif model_name == 'hybrid_edge_aligned':
        return HybridEdgeConvAlignedModel(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Supported models are: cnn, bilstm, transformer, transformer_gelu, transformer_aligned, hybrid_grf, hybrid_edge, hybrid_gat, hybrid_gcn_res, hybrid_gcn_edge_parallel, hybrid_grf_aligned, hybrid_edge_aligned.")
