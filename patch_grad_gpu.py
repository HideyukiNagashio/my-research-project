with open("scripts/visualize_gradient.py", "r") as f:
    content = f.read()

old_func = """    dynamics_maps = np.zeros((in_dim, seq_len, seq_len))
    
    # 1. Forward pass for the entire batch
    if timers is not None:
        t_fwd_start = time.time()
    outputs = model(input_data)  # Shape: (batch_size, seq_len, out_dim)
    if timers is not None:
        timers['Forward'] += time.time() - t_fwd_start
        timers['forward_calls'] += 1
        
    for x in range(seq_len):
        if input_data.grad is not None:
            input_data.grad.zero_()
        model.zero_grad()
        
        # Taking the sum over the batch computes independent gradients perfectly 
        # because the forward pass across samples in a batch is completely independent.
        score = outputs[:, x, out_col].sum()
        
        # 2. Backward pass. Retain graph for all steps except the last one to release memory.
        is_last = (x == seq_len - 1)
        if timers is not None:
            t_bwd_start = time.time()
        score.backward(retain_graph=not is_last)
        if timers is not None:
            timers['Backward'] += time.time() - t_bwd_start
            
        if input_data.grad is not None:
            if timers is not None:
                t_agg_start = time.time()
            
            # input_data.grad shape is (batch_size, seq_len, in_dim)
            # We want to sum the absolute gradients across the batch
            grad_all_sum = np.sum(np.abs(input_data.grad.cpu().numpy()), axis=0)  # Shape: (seq_len, in_dim)
            dynamics_maps[:, :, x] = grad_all_sum.T
            
            if timers is not None:
                timers['Aggregation'] += time.time() - t_agg_start"""

new_func = """    import torch
    
    # Pre-allocate result on GPU to avoid CPU synchronization in the loop
    dynamics_maps_gpu = torch.zeros((in_dim, seq_len, seq_len), device=input_data.device)
    
    # 1. Forward pass for the entire batch
    if timers is not None:
        t_fwd_start = time.time()
    outputs = model(input_data)  # Shape: (batch_size, seq_len, out_dim)
    if timers is not None:
        timers['Forward'] += time.time() - t_fwd_start
        timers['forward_calls'] += 1
        
    for x in range(seq_len):
        if input_data.grad is not None:
            input_data.grad.zero_()
        model.zero_grad()
        
        # Taking the sum over the batch computes independent gradients perfectly 
        # because the forward pass across samples in a batch is completely independent.
        score = outputs[:, x, out_col].sum()
        
        # 2. Backward pass. Retain graph for all steps except the last one to release memory.
        is_last = (x == seq_len - 1)
        if timers is not None:
            t_bwd_start = time.time()
        score.backward(retain_graph=not is_last)
        if timers is not None:
            timers['Backward'] += time.time() - t_bwd_start
            
        if input_data.grad is not None:
            if timers is not None:
                t_agg_start = time.time()
            
            # input_data.grad shape is (batch_size, seq_len, in_dim)
            # We sum the absolute gradients across the batch entirely on the GPU
            grad_all_sum = input_data.grad.abs().sum(dim=0)  # Shape: (seq_len, in_dim) on GPU
            dynamics_maps_gpu[:, :, x] = grad_all_sum.T
            
            if timers is not None:
                timers['Aggregation'] += time.time() - t_agg_start
                
    # Final single transfer to CPU memory
    dynamics_maps = dynamics_maps_gpu.cpu().numpy()"""

content = content.replace(old_func, new_func)
with open("scripts/visualize_gradient.py", "w") as f:
    f.write(content)
