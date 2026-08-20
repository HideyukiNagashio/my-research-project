with open("scripts/visualize_attention.py", "r") as f:
    content = f.read()

# 1. Modify extract_attention_maps to return PyTorch tensors
old_extract = """            if attn_weights.dim() == 3:
                # Add head dimension: (Batch, SeqLen, SeqLen) -> (Batch, 1, SeqLen, SeqLen)
                attn_weights = attn_weights.unsqueeze(1)
                
            attention_maps.append(attn_weights.cpu().numpy())
            
        # 3. Final regression head
        output = model.fc(current_features)
        
    return output.cpu().numpy(), attention_maps"""

new_extract = """            if attn_weights.dim() == 3:
                # Add head dimension: (Batch, SeqLen, SeqLen) -> (Batch, 1, SeqLen, SeqLen)
                attn_weights = attn_weights.unsqueeze(1)
                
            # Keep attention maps on GPU as PyTorch tensors for fast Rollout calculation
            attention_maps.append(attn_weights)
            
        # 3. Final regression head
        output = model.fc(current_features)
        
    return output.cpu().numpy(), attention_maps"""

content = content.replace(old_extract, new_extract)

# 2. Modify compute_rollout_batch to use PyTorch
old_rollout = """def calculate_attention_rollout(attention_maps: list, head_idx: str = "mean") -> np.ndarray:
    \"\"\"
    Computes cumulative Attention Rollout considering residual connections and layer product.
    R_l = \hat{A}_l @ R_{l-1}, where \hat{A}_l = A_l + I (normalized).
    
    Args:
        attention_maps: List of length num_layers. Each item is shape (Batch, nhead, SeqLen, SeqLen).
        head_idx: 'mean' or specific head index.
        
    Returns:
        Rollout matrix of shape (Batch, SeqLen, SeqLen) (Batch, Query, Key).
    \"\"\"
    num_layers = len(attention_maps)
    batch_size = attention_maps[0].shape[0]
    seq_len = attention_maps[0].shape[-1]
    
    # Initialize rollout R_0 as identity matrix (Batch, SeqLen, SeqLen)
    R = np.tile(np.eye(seq_len), (batch_size, 1, 1))
    
    for layer_idx in range(num_layers):
        layer_map = attention_maps[layer_idx] # (Batch, nhead, SeqLen, SeqLen)
        
        # 1. Average or select heads
        if head_idx == "mean":
            A = np.mean(layer_map, axis=1) # (Batch, SeqLen, SeqLen)
        else:
            A = layer_map[:, int(head_idx)] # (Batch, SeqLen, SeqLen)
            
        # 2. Add residual connection: \hat{A} = A + I
        I = np.tile(np.eye(seq_len), (batch_size, 1, 1))
        A_hat = A + I
        
        # 3. Row normalization: make rows sum to 1. Query is axis=1, Key is axis=2.
        A_hat = A_hat / A_hat.sum(axis=-1, keepdims=True)
        
        # 4. Multiply with cumulative rollout: R_l = A_hat_l @ R_{l-1}
        R = np.matmul(A_hat, R)
        
    return R"""

new_rollout = """def calculate_attention_rollout(attention_maps: list, head_idx: str = "mean"):
    \"\"\"
    Computes cumulative Attention Rollout considering residual connections and layer product.
    Optimized for PyTorch GPU execution to prevent CPU bottlenecks.
    
    Args:
        attention_maps: List of PyTorch tensors, each shape (Batch, nhead, SeqLen, SeqLen) on GPU.
        head_idx: 'mean' or specific head index.
        
    Returns:
        Rollout matrix of shape (Batch, SeqLen, SeqLen) (Batch, Query, Key) as a PyTorch tensor on GPU.
    \"\"\"
    import torch
    num_layers = len(attention_maps)
    batch_size = attention_maps[0].shape[0]
    seq_len = attention_maps[0].shape[-1]
    device = attention_maps[0].device
    
    # Initialize rollout R_0 as identity matrix (Batch, SeqLen, SeqLen) on GPU
    I = torch.eye(seq_len, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    R = I.clone()
    
    for layer_idx in range(num_layers):
        layer_map = attention_maps[layer_idx] # (Batch, nhead, SeqLen, SeqLen)
        
        # 1. Average or select heads
        if head_idx == "mean":
            A = torch.mean(layer_map, dim=1) # (Batch, SeqLen, SeqLen)
        else:
            A = layer_map[:, int(head_idx)] # (Batch, SeqLen, SeqLen)
            
        # 2. Add residual connection: \hat{A} = A + I
        A_hat = A + I
        
        # 3. Row normalization: make rows sum to 1. Query is axis=1, Key is axis=2.
        A_hat = A_hat / A_hat.sum(dim=-1, keepdim=True)
        
        # 4. Multiply with cumulative rollout: R_l = A_hat_l @ R_{l-1}
        # torch.bmm is extremely fast for batched matrix multiplication on GPU
        R = torch.bmm(A_hat, R)
        
    return R"""

content = content.replace(old_rollout, new_rollout)

# 3. Convert rollout and map arrays back to numpy right after calculation in the main loop
old_calc = """            _, batch_attn_maps = extract_attention_maps(model, inputs_batch, device)
            batch_rollout = compute_rollout_batch(batch_attn_maps, head_idx="mean")"""

new_calc = """            _, batch_attn_maps_gpu = extract_attention_maps(model, inputs_batch, device)
            batch_rollout_gpu = calculate_attention_rollout(batch_attn_maps_gpu, head_idx=args.head_idx)
            
            # Transfer to CPU ONLY once per batch after all heavy calculations are done
            batch_attn_maps = [m.cpu().numpy() for m in batch_attn_maps_gpu]
            batch_rollout = batch_rollout_gpu.cpu().numpy()"""
            
content = content.replace("            batch_rollout = compute_rollout_batch(batch_attn_maps, head_idx=\"mean\")", new_calc.split("\\n")[-1].strip() + "\\n" + "            batch_rollout = batch_rollout_gpu.cpu().numpy()")

# I need a robust replace for the exact loop computation:
old_loop_calc = """            _, batch_attn_maps = extract_attention_maps(model, inputs_batch, device)
            batch_rollout = calculate_attention_rollout(batch_attn_maps, head_idx=args.head_idx)"""
            
new_loop_calc = """            _, batch_attn_maps_gpu = extract_attention_maps(model, inputs_batch, device)
            batch_rollout_gpu = calculate_attention_rollout(batch_attn_maps_gpu, head_idx=args.head_idx)
            
            # Transfer to CPU ONLY once per batch after all heavy calculations are done
            batch_attn_maps = [m.cpu().numpy() for m in batch_attn_maps_gpu]
            batch_rollout = batch_rollout_gpu.cpu().numpy()"""

content = content.replace(old_loop_calc, new_loop_calc)


# Wait, in the code, the function is called calculate_attention_rollout, but my previous patch may have called it compute_rollout_batch? 
# Ah, the original code had calculate_attention_rollout, but my patch had compute_rollout_batch?
# Let's check what I replaced it with in the previous python script!
