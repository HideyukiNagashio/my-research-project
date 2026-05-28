#!/usr/bin/env python3
"""
Attention Map (Self-Attention Weights) Extraction and Visualization Script
for TimeSeriesTransformer GRF Estimation Model.
"""

import os
import sys
import json
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib
# Set non-interactive backend if no display is available or --no_show is passed
if 'DISPLAY' not in os.environ or '--no_show' in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch

# Add the project root to the python path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Extract and visualize Transformer Attention Maps")
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Path to the experiment output directory (e.g. outputs/experiments/transformer_grf_single_weighted_transformer_...)"
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6],
        help="Fold number to visualize (1-6)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "aggregate"],
        help="Visualization mode: 'single' for a specific sample, 'aggregate' for average attention across samples"
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Index of the test sample to visualize (only used in 'single' mode)"
    )
    parser.add_argument(
        "--subject_name",
        type=str,
        default=None,
        help="Filter samples by subject name (e.g., 'Sub001')"
    )
    parser.add_argument(
        "--condition_name",
        type=str,
        default=None,
        help="Filter samples by condition name (e.g., 'NormalWalk')"
    )
    parser.add_argument(
        "--head_idx",
        type=str,
        default="mean",
        help="Attention head index to plot (e.g. '0', '1', '2', '3' or 'mean' for average over heads)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save output plots (defaults to <exp_dir>/attention_plots/)"
    )
    parser.add_argument(
        "--save_format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Image format to save"
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Do not display the plot (save only)"
    )
    return parser.parse_args()


def extract_attention_maps(model, x, device='cpu'):
    """
    Extract self-attention weights from the TimeSeriesTransformer model.
    Handles different PyTorch versions and encoder setups.
    
    Args:
        model: TimeSeriesTransformer model instance
        x: Input tensor of shape (Batch, SeqLen, InputDim)
        device: Device to run the forward pass
        
    Returns:
        output: Predicted outputs of shape (Batch, SeqLen, OutputDim)
        attention_maps: List of numpy arrays, one per layer.
                        Each array has shape (Batch, nhead, SeqLen, SeqLen).
    """
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        
        # 1. Embedding & Positional Encoding
        x_emb = model.embedding(x) * math.sqrt(model.d_model)
        x_pe = model.pos_encoder(x_emb)
        
        # 2. Sequential execution of Transformer Encoder layers
        current_features = x_pe
        attention_maps = []
        
        for layer in model.transformer_encoder.layers:
            # Check if self_attn supports average_attn_weights parameter (PyTorch >= 1.11)
            import inspect
            sig = inspect.signature(layer.self_attn.forward)
            has_average_attn_weights = 'average_attn_weights' in sig.parameters
            
            norm_first = getattr(layer, "norm_first", False)
            h = current_features
            
            if norm_first:
                h_norm = layer.norm1(h)
                if has_average_attn_weights:
                    attn_output, attn_weights = layer.self_attn(
                        h_norm, h_norm, h_norm,
                        need_weights=True,
                        average_attn_weights=False
                    )
                else:
                    attn_output, attn_weights = layer.self_attn(
                        h_norm, h_norm, h_norm,
                        need_weights=True
                    )
                h = h + layer.dropout1(attn_output)
                
                h_norm2 = layer.norm2(h)
                if hasattr(layer, "_ff_block"):
                    ff_out = layer._ff_block(h_norm2)
                else:
                    ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(h_norm2))))
                current_features = h + layer.dropout2(ff_out)
            else:
                if has_average_attn_weights:
                    attn_output, attn_weights = layer.self_attn(
                        h, h, h,
                        need_weights=True,
                        average_attn_weights=False
                    )
                else:
                    attn_output, attn_weights = layer.self_attn(
                        h, h, h,
                        need_weights=True
                    )
                h = layer.norm1(h + layer.dropout1(attn_output))
                
                if hasattr(layer, "_ff_block"):
                    ff_out = layer._ff_block(h)
                else:
                    ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(h))))
                current_features = layer.norm2(h + layer.dropout2(ff_out))
                
            # If PyTorch doesn't support returning non-averaged weights, shape is (Batch, SeqLen, SeqLen)
            if attn_weights.dim() == 3:
                # Add head dimension: (Batch, SeqLen, SeqLen) -> (Batch, 1, SeqLen, SeqLen)
                attn_weights = attn_weights.unsqueeze(1)
                
            attention_maps.append(attn_weights.cpu().numpy())
            
        # 3. Final regression head
        output = model.fc(current_features)
        
    return output.cpu().numpy(), attention_maps


def load_data_and_model(exp_dir, fold, device):
    """
    Load model configuration, model weights, and corresponding test arrays.
    """
    # 1. Load config
    config_path = os.path.join(exp_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, "r") as f:
        config = json.load(f)
        
    # 2. Get input/output dimensions
    input_type = config.get("input_type", "single_leg")
    target_type = config.get("target_type", "grf_only")
    
    if input_type == 'single_leg': in_dim = 14
    elif input_type == 'bilateral': in_dim = 28
    elif input_type == 'pressure_single': in_dim = 8
    elif input_type == 'pressure_bilateral': in_dim = 16
    elif input_type == 'imu_single': in_dim = 6
    elif input_type == 'imu_bilateral': in_dim = 12
    else: raise ValueError(f"Unknown input_type {input_type}")

    if target_type == 'all': out_dim = 12
    elif target_type == 'angles_only': out_dim = 9
    elif target_type == 'grf_only': out_dim = 3
    else: raise ValueError(f"Unknown target_type {target_type}")

    # 3. Re-instantiate the model
    model_kwargs = {
        'input_dim': in_dim,
        'output_dim': out_dim,
        'dropout_prob': config.get("dropout", 0.1),
        'd_model': config.get("d_model", 128),
        'nhead': config.get("nhead", 4),
        'num_layers': config.get("num_layers", 3),
        'dim_feedforward': config.get("dim_feedforward", 256)
    }
    
    model = get_model("transformer", **model_kwargs)
    
    # 4. Load weights
    model_path = os.path.join(exp_dir, f"best_model_fold{fold}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"Loaded model: {model_path}")
    
    # 5. Load test data files
    inputs = np.load(os.path.join(exp_dir, f"inputs_fold{fold}.npy"))
    preds = np.load(os.path.join(exp_dir, f"preds_fold{fold}.npy"))
    targets = np.load(os.path.join(exp_dir, f"targets_fold{fold}.npy"))
    
    # Load meta CSV if available
    meta_path = os.path.join(exp_dir, f"sample_meta_fold{fold}.csv")
    meta_df = None
    if os.path.exists(meta_path):
        meta_df = pd.read_csv(meta_path)
        print(f"Loaded sample metadata: {meta_path} ({len(meta_df)} samples)")
    else:
        print(f"Warning: Metadata file not found at {meta_path}")
        
    # Feature/Target names
    feature_names = []
    feature_path = os.path.join(exp_dir, "feature_names.json")
    if os.path.exists(feature_path):
        with open(feature_path, "r") as f:
            feature_names = json.load(f)
            
    target_names = []
    target_path = os.path.join(exp_dir, "target_names.json")
    if os.path.exists(target_path):
        with open(target_path, "r") as f:
            target_names = json.load(f)

    return model, inputs, preds, targets, meta_df, feature_names, target_names


def plot_single_sample(sample_idx, inputs, preds, targets, attention_maps, feature_names, target_names, head_idx, save_path):
    """
    Visualize a single stride cycle:
    1. Attention Map 2D Heatmaps for each layer.
    2. Overlay of Ground Truth vs Predicted GRF.
    3. Representative plantar pressure sensor values.
    4. 1D Attention Profile (average attention directed to each timestep).
    """
    num_layers = len(attention_maps)
    num_timesteps = inputs.shape[1]
    
    layer_maps = []
    for layer_idx in range(num_layers):
        attn_map = attention_maps[layer_idx][sample_idx] # (nhead, SeqLen, SeqLen)
        
        if head_idx == "mean":
            layer_maps.append(np.mean(attn_map, axis=0))
        else:
            h_idx = int(head_idx)
            if h_idx >= attn_map.shape[0]:
                print(f"Warning: Requested head {h_idx} is out of bounds. Using average.")
                layer_maps.append(np.mean(attn_map, axis=0))
            else:
                layer_maps.append(attn_map[h_idx])
                
    attention_profiles = []
    for layer_idx in range(num_layers):
        profile = np.mean(layer_maps[layer_idx], axis=0)
        attention_profiles.append(profile)

    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    gs = gridspec.GridSpec(4, 3, figure=fig, height_ratios=[1.2, 1.0, 1.0, 1.0])
    
    # --- Row 1: Attention Map Heatmaps ---
    for layer_idx in range(num_layers):
        ax = fig.add_subplot(gs[0, layer_idx])
        sns.heatmap(
            layer_maps[layer_idx].T,
            cmap="viridis",
            ax=ax,
            cbar=True,
            square=True,
            xticklabels=20,
            yticklabels=20
        )
        ax.set_title(f"Layer {layer_idx + 1} Attention Map (Head: {head_idx})", fontsize=12, fontweight='bold')
        ax.set_ylabel("Key (Attended Timestep)", fontsize=10)
        ax.invert_yaxis()

    # --- Row 2: vertical GRF (Fz) ---
    ax_fz = fig.add_subplot(gs[1, :])
    time_steps = np.arange(num_timesteps)
    
    fz_idx = target_names.index("Fz") if "Fz" in target_names else min(2, targets.shape[2] - 1)
    
    ax_fz.plot(time_steps, targets[sample_idx, :, fz_idx], 'k-', label="Ground Truth (Fz)", linewidth=2.5)
    ax_fz.plot(time_steps, preds[sample_idx, :, fz_idx], 'r--', label="Predicted (Fz)", linewidth=2.0)
    ax_fz.set_ylabel("Vertical GRF (Fz) [N]", color="k", fontsize=11, fontweight='bold')
    ax_fz.tick_params(axis='y', labelcolor="k")
    ax_fz.grid(True, linestyle=":", alpha=0.6)
    
    ax_fz_twin = ax_fz.twinx()
    mean_profile = np.mean(attention_profiles, axis=0)
    ax_fz_twin.fill_between(time_steps, 0, mean_profile, color="tab:blue", alpha=0.25, label="Mean Attention Weight")
    ax_fz_twin.plot(time_steps, mean_profile, color="tab:blue", linewidth=1.5, alpha=0.7)
    ax_fz_twin.set_ylabel("Attention Weight (Key Mean)", color="tab:blue", fontsize=11, fontweight='bold')
    ax_fz_twin.tick_params(axis='y', labelcolor="tab:blue")
    
    lines_1, labels_1 = ax_fz.get_legend_handles_labels()
    lines_2, labels_2 = ax_fz_twin.get_legend_handles_labels()
    ax_fz.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    ax_fz.set_title("Vertical GRF (Fz) vs. Attention Profile Over Time", fontsize=12, fontweight='bold')
    ax_fz.set_xlim(0, num_timesteps - 1)

    # --- Row 3: Shear GRFs (Fx, Fy) ---
    ax_shear = fig.add_subplot(gs[2, :])
    fx_idx = target_names.index("Fx") if "Fx" in target_names else 0
    fy_idx = target_names.index("Fy") if "Fy" in target_names else 1
    
    if targets.shape[2] > fx_idx:
        ax_shear.plot(time_steps, targets[sample_idx, :, fx_idx], 'g-', label="Ground Truth (Fx)", alpha=0.7)
        ax_shear.plot(time_steps, preds[sample_idx, :, fx_idx], 'g--', label="Predicted (Fx)", alpha=0.9)
    if targets.shape[2] > fy_idx:
        ax_shear.plot(time_steps, targets[sample_idx, :, fy_idx], 'b-', label="Ground Truth (Fy)", alpha=0.7)
        ax_shear.plot(time_steps, preds[sample_idx, :, fy_idx], 'b--', label="Predicted (Fy)", alpha=0.9)
        
    ax_shear.set_ylabel("Shear GRF (Fx, Fy) [N]", fontsize=11, fontweight='bold')
    ax_shear.grid(True, linestyle=":", alpha=0.6)
    ax_shear.legend(loc="upper right")
    ax_shear.set_title("Shear GRFs (Fx, Fy) Prediction", fontsize=12, fontweight='bold')
    ax_shear.set_xlim(0, num_timesteps - 1)

    # --- Row 4: Plantar Pressure & Layer-wise Attention Profiles ---
    ax_press = fig.add_subplot(gs[3, :2])
    press_cols = [col for col in feature_names if col.startswith("P")]
    if not press_cols:
        press_cols = [feature_names[i] for i in range(min(8, len(feature_names)))]
        
    for col in press_cols[:4]:
        col_idx = feature_names.index(col)
        ax_press.plot(time_steps, inputs[sample_idx, :, col_idx], label=f"Pressure {col}", alpha=0.7)
        
    ax_press_twin = ax_press.twinx()
    for col in press_cols[4:8]:
        col_idx = feature_names.index(col)
        ax_press_twin.plot(time_steps, inputs[sample_idx, :, col_idx], ':', label=f"Pressure {col}", alpha=0.7)
        
    ax_press.set_ylabel("Forefoot Pressure [V/Scale]", fontsize=11)
    ax_press_twin.set_ylabel("Heel Pressure [V/Scale]", fontsize=11)
    ax_press.grid(True, linestyle=":", alpha=0.6)
    
    h1, l1 = ax_press.get_legend_handles_labels()
    h2, l2 = ax_press_twin.get_legend_handles_labels()
    ax_press.legend(h1 + h2, l1 + l2, loc="upper right")
    ax_press.set_title("Plantar Pressure Waveforms (Heel-Strike to Toe-Off check)", fontsize=12, fontweight='bold')
    ax_press.set_xlabel("Time step", fontsize=10)
    ax_press.set_xlim(0, num_timesteps - 1)
    
    ax_layer_attn = fig.add_subplot(gs[3, 2])
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for layer_idx in range(num_layers):
        ax_layer_attn.plot(
            time_steps,
            attention_profiles[layer_idx],
            label=f"Layer {layer_idx + 1}",
            color=colors[layer_idx % len(colors)],
            linewidth=2.0
        )
    ax_layer_attn.set_title("Attention Profile per Layer", fontsize=12, fontweight='bold')
    ax_layer_attn.set_xlabel("Time step", fontsize=10)
    ax_layer_attn.set_ylabel("Key-wise Attention Mean", fontsize=10)
    ax_layer_attn.grid(True, linestyle=":", alpha=0.6)
    ax_layer_attn.legend(loc="upper right")
    ax_layer_attn.set_xlim(0, num_timesteps - 1)
    
    plt.suptitle(f"Biomechanical Attention Analysis (Sample Index: {sample_idx})", fontsize=16, fontweight='bold', y=0.99)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Saved single sample attention plot to: {save_path}")
    plt.close()


def plot_aggregate_attention(inputs, preds, targets, attention_maps, target_names, head_idx, save_path, filter_desc=""):
    """
    Plot aggregated attention maps and attention profiles across multiple strides to show general model behavior.
    """
    num_samples = inputs.shape[0]
    num_layers = len(attention_maps)
    num_timesteps = inputs.shape[1]
    
    aggregated_maps = []
    for layer_idx in range(num_layers):
        layer_map = attention_maps[layer_idx]
        if head_idx == "mean":
            agg = np.mean(np.mean(layer_map, axis=1), axis=0)
        else:
            h_idx = int(head_idx)
            agg = np.mean(layer_map[:, h_idx], axis=0)
        aggregated_maps.append(agg)
        
    attention_profiles = []
    for layer_idx in range(num_layers):
        if head_idx == "mean":
            h_map = np.mean(attention_maps[layer_idx], axis=1)
        else:
            h_map = attention_maps[layer_idx][:, int(head_idx)]
        
        profiles = np.mean(h_map, axis=1)
        attention_profiles.append(profiles)
        
    fz_idx = target_names.index("Fz") if "Fz" in target_names else min(2, targets.shape[2] - 1)
    fz_targets = targets[:, :, fz_idx]
    fz_preds = preds[:, :, fz_idx]
    
    mean_fz_target = np.mean(fz_targets, axis=0)
    std_fz_target = np.std(fz_targets, axis=0)
    mean_fz_pred = np.mean(fz_preds, axis=0)
    std_fz_pred = np.std(fz_preds, axis=0)
    
    time_steps = np.arange(num_timesteps)
    
    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1.2, 1.0, 1.0])
    
    # --- Row 1: Aggregated 2D Attention Maps ---
    for layer_idx in range(num_layers):
        ax = fig.add_subplot(gs[0, layer_idx])
        sns.heatmap(
            aggregated_maps[layer_idx].T,
            cmap="viridis",
            ax=ax,
            cbar=True,
            square=True,
            xticklabels=20,
            yticklabels=20
        )
        ax.set_title(f"Aggregated Layer {layer_idx + 1} Attention Map (Head: {head_idx})", fontsize=11, fontweight='bold')
        ax.set_xlabel("Query (Source Timestep)", fontsize=9)
        ax.set_ylabel("Key (Attended Timestep)", fontsize=9)
        ax.invert_yaxis()

    # --- Row 2: Average Fz Profile with Attention Overlay ---
    ax_fz = fig.add_subplot(gs[1, :])
    ax_fz.plot(time_steps, mean_fz_target, 'k-', label="Average Ground Truth Fz", linewidth=2.5)
    ax_fz.fill_between(time_steps, mean_fz_target - std_fz_target, mean_fz_target + std_fz_target, color='black', alpha=0.15)
    
    ax_fz.plot(time_steps, mean_fz_pred, 'r--', label="Average Predicted Fz", linewidth=2.0)
    ax_fz.fill_between(time_steps, mean_fz_pred - std_fz_pred, mean_fz_pred + std_fz_pred, color='red', alpha=0.1)
    
    ax_fz.set_ylabel("Vertical GRF (Fz) [N]", color="k", fontsize=11, fontweight='bold')
    ax_fz.tick_params(axis='y', labelcolor="k")
    ax_fz.grid(True, linestyle=":", alpha=0.6)
    
    all_profiles_flat = np.concatenate([p[np.newaxis, :, :] for p in attention_profiles], axis=0)
    global_mean_profile = np.mean(np.mean(all_profiles_flat, axis=0), axis=0)
    global_std_profile = np.std(np.mean(all_profiles_flat, axis=0), axis=0)
    
    ax_fz_twin = ax_fz.twinx()
    ax_fz_twin.plot(time_steps, global_mean_profile, color="tab:blue", linewidth=2.0, label="Global Mean Attention")
    ax_fz_twin.fill_between(
        time_steps,
        np.maximum(0, global_mean_profile - global_std_profile),
        global_mean_profile + global_std_profile,
        color="tab:blue",
        alpha=0.2
    )
    ax_fz_twin.set_ylabel("Attention Weight (Key Mean)", color="tab:blue", fontsize=11, fontweight='bold')
    ax_fz_twin.tick_params(axis='y', labelcolor="tab:blue")
    
    h1, l1 = ax_fz.get_legend_handles_labels()
    h2, l2 = ax_fz_twin.get_legend_handles_labels()
    ax_fz.legend(h1 + h2, l1 + l2, loc="upper right")
    ax_fz.set_title(f"Aggregated Fz Waveforms vs. Global Attention Profile (N={num_samples})", fontsize=12, fontweight='bold')
    ax_fz.set_xlim(0, num_timesteps - 1)

    # --- Row 3: Layer-wise Aggregated Profiles ---
    ax_layers = fig.add_subplot(gs[2, :])
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for layer_idx in range(num_layers):
        layer_means = np.mean(attention_profiles[layer_idx], axis=0)
        layer_stds = np.std(attention_profiles[layer_idx], axis=0)
        
        ax_layers.plot(
            time_steps,
            layer_means,
            label=f"Layer {layer_idx + 1}",
            color=colors[layer_idx % len(colors)],
            linewidth=2.0
        )
        ax_layers.fill_between(
            time_steps,
            np.maximum(0, layer_means - layer_stds),
            layer_means + layer_stds,
            color=colors[layer_idx % len(colors)],
            alpha=0.1
        )
        
    ax_layers.set_title("Aggregated Attention Profile per Layer (Mean ± Std)", fontsize=12, fontweight='bold')
    ax_layers.set_xlabel("Time step", fontsize=10)
    ax_layers.set_ylabel("Key-wise Attention Mean", fontsize=10)
    ax_layers.grid(True, linestyle=":", alpha=0.6)
    ax_layers.legend(loc="upper right")
    ax_layers.set_xlim(0, num_timesteps - 1)
    
    title_str = f"Aggregated Biomechanical Attention Analysis (N={num_samples} Strides)"
    if filter_desc:
        title_str += f" - {filter_desc}"
    plt.suptitle(title_str, fontsize=16, fontweight='bold', y=0.99)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Saved aggregated attention plot to: {save_path}")
    plt.close()


def main():
    args = parse_args()
    
    # Establish device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load config, data, model
    try:
        model, inputs, preds, targets, meta_df, feature_names, target_names = load_data_and_model(
            args.exp_dir, args.fold, device
        )
    except Exception as e:
        print(f"Error loading files: {e}")
        print("Please check if the '--exp_dir' and '--fold' parameters are correct and files exist on the path.")
        sys.exit(1)
        
    # Convert numpy inputs to PyTorch tensor
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    
    # 2. Extract attention maps
    print("Extracting attention weights (running model forward pass)...")
    _, attention_maps = extract_attention_maps(model, inputs_tensor, device)
    
    # 3. Handle output directory
    if args.output_dir is None:
        output_dir = os.path.join(args.exp_dir, "attention_plots")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 4. Filter samples by metadata if specified
    sample_indices = np.arange(len(inputs))
    filter_desc = ""
    
    if meta_df is not None:
        filtered_mask = np.ones(len(meta_df), dtype=bool)
        if args.subject_name:
            filtered_mask &= (meta_df['subject_name'] == args.subject_name)
            filter_desc += f"Subject: {args.subject_name} "
        if args.condition_name:
            filtered_mask &= (meta_df['condition_name'] == args.condition_name)
            filter_desc += f"Condition: {args.condition_name}"
            
        filtered_indices = np.where(filtered_mask)[0]
        if len(filtered_indices) == 0:
            print(f"Warning: No samples match criteria (subject={args.subject_name}, condition={args.condition_name}). Ignoring filters.")
        else:
            sample_indices = filtered_indices
            print(f"Filtered to {len(sample_indices)} samples matching criteria.")
            if args.mode == "single" and args.sample_idx not in sample_indices:
                args.sample_idx = sample_indices[0]
                print(f"Selected sample_idx {args.sample_idx} as it matches the filter criteria.")

    # 5. Run Visualization
    if args.mode == "single":
        if args.sample_idx >= len(inputs) or args.sample_idx < 0:
            print(f"Error: sample_idx {args.sample_idx} is out of bounds (0 to {len(inputs) - 1})")
            sys.exit(1)
            
        meta_suffix = f"_fold{args.fold}_sample{args.sample_idx}"
        if meta_df is not None and args.sample_idx < len(meta_df):
            sub_name = meta_df.loc[args.sample_idx, 'subject_name']
            cond_name = meta_df.loc[args.sample_idx, 'condition_name']
            meta_suffix += f"_{sub_name}_{cond_name}"
            print(f"Visualizing Sample {args.sample_idx} (Subject: {sub_name}, Condition: {cond_name})")
            
        save_path = os.path.join(output_dir, f"attention_single{meta_suffix}.{args.save_format}")
        plot_single_sample(
            args.sample_idx, inputs, preds, targets, attention_maps,
            feature_names, target_names, args.head_idx, save_path
        )
        
    elif args.mode == "aggregate":
        agg_inputs = inputs[sample_indices]
        agg_preds = preds[sample_indices]
        agg_targets = targets[sample_indices]
        agg_maps = [m[sample_indices] for m in attention_maps]
        
        filter_suffix = ""
        if args.subject_name:
            filter_suffix += f"_{args.subject_name}"
        if args.condition_name:
            filter_suffix += f"_{args.condition_name}"
            
        save_path = os.path.join(output_dir, f"attention_aggregate_fold{args.fold}{filter_suffix}.{args.save_format}")
        plot_aggregate_attention(
            agg_inputs, agg_preds, agg_targets, agg_maps,
            target_names, args.head_idx, save_path, filter_desc
        )

    # 6. Show plot if interactive and no_show is False
    if not args.no_show:
        try:
            plt.show()
        except Exception:
            print("No graphic display detected or backend not available. Plots were saved to file.")


if __name__ == "__main__":
    main()
