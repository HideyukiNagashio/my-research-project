import os
import json

def count_cnn_params(input_dim, hidden_dim, kernel_size, output_dim):
    # features Block 1
    c1_w = hidden_dim * input_dim * kernel_size
    c1_b = hidden_dim
    bn1_w = hidden_dim
    bn1_b = hidden_dim
    
    # Block 2
    c2_w = (hidden_dim * 2) * hidden_dim * 3
    c2_b = hidden_dim * 2
    bn2_w = hidden_dim * 2
    bn2_b = hidden_dim * 2
    
    # Block 3
    c3_w = (hidden_dim * 2) * (hidden_dim * 2) * 3
    c3_b = hidden_dim * 2
    bn3_w = hidden_dim * 2
    bn3_b = hidden_dim * 2
    
    # Regressor Head
    fc1_w = 128 * (hidden_dim * 2)
    fc1_b = 128
    fc2_w = output_dim * 128
    fc2_b = output_dim
    
    trainable = (
        c1_w + c1_b + bn1_w + bn1_b +
        c2_w + c2_b + bn2_w + bn2_b +
        c3_w + c3_b + bn3_w + bn3_b +
        fc1_w + fc1_b + fc2_w + fc2_b
    )
    
    # Non-trainable BatchNorm parameters: running_mean, running_var, and num_batches_tracked (1 per BN)
    non_trainable = 2 * (hidden_dim + (hidden_dim * 2) + (hidden_dim * 2)) + 3
    
    return trainable, trainable + non_trainable

def count_bilstm_params(input_dim, hidden_dim, num_layers, output_dim):
    lstm_params = 0
    for layer in range(num_layers):
        if layer == 0:
            I = input_dim
        else:
            I = hidden_dim * 2
        # forward
        lstm_params += 4 * hidden_dim * (I + hidden_dim + 2)
        # backward
        lstm_params += 4 * hidden_dim * (I + hidden_dim + 2)
        
    ln_w = hidden_dim * 2
    ln_b = hidden_dim * 2
    
    fc1_w = 128 * (hidden_dim * 2)
    fc1_b = 128
    fc2_w = output_dim * 128
    fc2_b = output_dim
    
    trainable = lstm_params + ln_w + ln_b + fc1_w + fc1_b + fc2_w + fc2_b
    return trainable, trainable

def count_transformer_params(input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim):
    emb_w = d_model * input_dim
    emb_b = d_model
    
    attn_in_proj = 3 * d_model * d_model + 3 * d_model
    attn_out_proj = d_model * d_model + d_model
    ff_linear1 = dim_feedforward * d_model + dim_feedforward
    ff_linear2 = d_model * dim_feedforward + d_model
    norm1 = 2 * d_model
    norm2 = 2 * d_model
    
    layer_params = attn_in_proj + attn_out_proj + ff_linear1 + ff_linear2 + norm1 + norm2
    encoder_params = layer_params * num_layers
    
    fc1_w = 64 * d_model
    fc1_b = 64
    fc2_w = output_dim * 64
    fc2_b = output_dim
    
    trainable = emb_w + emb_b + encoder_params + fc1_w + fc1_b + fc2_w + fc2_b
    return trainable, trainable

def main():
    experiments_dir = "outputs/experiments"
    if not os.path.exists(experiments_dir):
        print(f"Error: {experiments_dir} does not exist.")
        return

    subdirs = sorted(os.listdir(experiments_dir))
    
    print(f"=== Model Parameter Summary (5 Conditions) ===\n")
    for subdir in subdirs:
        subdir_path = os.path.join(experiments_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        config_path = os.path.join(subdir_path, "config.json")
        if not os.path.exists(config_path):
            continue
            
        with open(config_path, "r") as f:
            config = json.load(f)
            
        input_type = config.get("input_type", "single_leg")
        target_type = config.get("target_type", "grf_only")
        model_type = config.get("model_type")
        
        # Determine in_dim
        if input_type == 'single_leg': in_dim = 14
        elif input_type == 'bilateral': in_dim = 28
        elif input_type == 'pressure_single': in_dim = 8
        elif input_type == 'pressure_bilateral': in_dim = 16
        elif input_type == 'imu_single': in_dim = 6
        elif input_type == 'imu_bilateral': in_dim = 12
        else:
            continue

        # Determine out_dim
        if target_type == 'all': out_dim = 12
        elif target_type == 'angles_only': out_dim = 9
        elif target_type == 'grf_only': out_dim = 3
        else:
            continue

        if model_type == 'cnn':
            hidden_dim = config.get("hidden_dim", 256)
            kernel_size = config.get("kernel_size", 5)
            trainable, total = count_cnn_params(in_dim, hidden_dim, kernel_size, out_dim)
        elif model_type == 'bilstm':
            hidden_dim = config.get("hidden_dim", 256)
            num_layers = config.get("num_layers", 3)
            trainable, total = count_bilstm_params(in_dim, hidden_dim, num_layers, out_dim)
        elif model_type == 'transformer':
            d_model = config.get("d_model", 128)
            nhead = config.get("nhead", 4)
            num_layers = config.get("num_layers", 3)
            dim_feedforward = config.get("dim_feedforward", 256)
            trainable, total = count_transformer_params(in_dim, d_model, nhead, num_layers, dim_feedforward, out_dim)
        else:
            continue
            
        print(f"Condition Directory: {subdir}")
        print(f"  Model Type          : {model_type.upper()}")
        print(f"  Input Dimension     : {in_dim} ({input_type})")
        print(f"  Output Dimension    : {out_dim} ({target_type})")
        print(f"  Trainable Parameters: {trainable:,}")
        print(f"  Total Parameters    : {total:,}")
        print("-" * 50)

if __name__ == "__main__":
    main()
