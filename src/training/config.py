import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='6-Fold Cross Validation Training Pipeline')
    
    # --- Experiment ---
    parser.add_argument('--exp_name', type=str, default='baseline', help='Name of the experiment to save outputs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # --- Data Settings ---
    parser.add_argument('--data_dir', type=str, default='data/processed/cv', help='Path to 6-fold CV Data')
    parser.add_argument('--input_type', type=str, default='bilateral', 
                        choices=['single_leg', 'bilateral', 'pressure_single', 'pressure_bilateral', 'imu_single', 'imu_bilateral'],
                        help='Input feature selection')
    parser.add_argument('--target_type', type=str, default='all', 
                        choices=['all', 'angles_only', 'grf_only'],
                        help='Target feature selection')
    
    # --- Model Settings ---
    parser.add_argument('--model_type', type=str, default='cnn', 
                        choices=['cnn', 'bilstm', 'transformer'],
                        help='Model architecture to use')
    # Transformer ones
    parser.add_argument('--d_model', type=int, default=128, help='Transformer embedding dim')
    parser.add_argument('--nhead', type=int, default=4, help='Transformer num heads')
    parser.add_argument('--dim_feedforward', type=int, default=256, help='Transformer FF dim')
    # CNN/LSTM ones
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden size for CNN/LSTM')
    parser.add_argument('--kernel_size', type=int, default=5, help='Kernel size for CNN')
    parser.add_argument('--num_layers', type=int, default=3, help='Num layers for Transformer/LSTM')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout probability')
    
    # --- Training Settings ---
    parser.add_argument('--epochs', type=int, default=200, help='Max number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size per step')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--factor', type=float, default=0.5, help='LR scheduler factor')
    
    # --- Loss Settings ---
    parser.add_argument('--use_weighted_loss', action='store_true', help='Use Weighted MSE Loss')
    # Fx, Fy, Fz weights setup; assuming target='all' order or target='grf_only' order
    # Note: Target specific weight shapes must be configured in run script based on target_type
    
    return parser
