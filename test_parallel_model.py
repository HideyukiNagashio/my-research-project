import torch
from src.models import get_model
from scripts.count_params import count_hybrid_gcn_edge_parallel_params

def test():
    print("Testing HybridGCNEdgeParallelModel...")
    # instantiate model
    model = get_model('hybrid_gcn_edge_parallel', input_dim=14, output_dim=3)
    
    # generate dummy input
    # x shape: (batch_size, seq_len, input_dim) -> (2, 200, 14)
    dummy_input = torch.randn(2, 200, 14)
    
    # forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape} (Expected: 2, 200, 3)")
    
    # check parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Actual Trainable Params: {trainable_params}")
    
    # check formula
    formula_trainable, _ = count_hybrid_gcn_edge_parallel_params(
        input_dim=14, num_layers=3, dim_feedforward=256, output_dim=3,
        gnn_out_dim=16, cnn_pool_dim=32, d_model=128, seq_len=200
    )
    print(f"Formula Trainable Params: {formula_trainable}")
    
    if trainable_params == formula_trainable:
        print("Parameter count matches!")
    else:
        print("Parameter count mismatch!")

if __name__ == '__main__':
    test()
