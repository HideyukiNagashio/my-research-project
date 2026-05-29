#!/usr/bin/env python3
"""
Gait Analysis Transformer Explainability Framework (XAI)
Extracts and visualizes self-attention maps, temporal bins, phase-to-phase attention, 
and cumulative attention rollouts from the TimeSeriesTransformer GRF Estimation Model.
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

# Default gait phase ratio (in % of stride cycle)
DEFAULT_GAIT_PHASES = {
    'LR':  (0.0, 10.0),   # Loading Response
    'MSt': (10.0, 30.0),  # Mid Stance
    'TSt': (30.0, 50.0),  # Terminal Stance
    'PSw': (50.0, 60.0),  # Pre Swing
    'ISw': (60.0, 75.0),  # Initial Swing
    'MSw': (75.0, 85.0),  # Mid Swing
    'TSw': (85.0, 100.0)  # Terminal Swing
}


def parse_args():
    parser = argparse.ArgumentParser(description="Gait Analysis Transformer Explainability Framework")
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Path to the experiment output directory"
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
        help="Visualization mode: 'single' for a specific sample, 'aggregate' for overall model evaluation"
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
        help="Attention head index to plot ('0', '1', '2', '3' or 'mean' for average over heads)"
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
        "--shared_cmap",
        action="store_true",
        help="Enable shared color map scale (vmin=0, vmax=global_vmax) across layers/heads/samples"
    )
    parser.add_argument(
        "--downsample_bins",
        type=int,
        default=None,
        help="Number of bins to downsample the attention map (e.g., 50, 25). Must divide sequence length."
    )
    parser.add_argument(
        "--gait_phases_json",
        type=str,
        default=None,
        help="Custom gait phase ratio JSON string to override default ratios"
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Do not display the plot (save only)"
    )
    return parser.parse_args()


# ==========================================
# Core Processing & Algorithm Functions
# ==========================================

def extract_attention_maps(model, x, device='cpu'):
    """
    Extract self-attention weights from the TimeSeriesTransformer model.
    
    Args:
        model: TimeSeriesTransformer model instance
        x: Input tensor of shape (Batch, SeqLen, InputDim)
        device: Device to run the forward pass
        
    Returns:
        output: Predicted outputs of shape (Batch, SeqLen, OutputDim)
        attention_maps: List of length num_layers.
                        Each array has shape (Batch, nhead, SeqLen, SeqLen) (Batch, Head, Query, Key).
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
                
            if attn_weights.dim() == 3:
                # Add head dimension: (Batch, SeqLen, SeqLen) -> (Batch, 1, SeqLen, SeqLen)
                attn_weights = attn_weights.unsqueeze(1)
                
            attention_maps.append(attn_weights.cpu().numpy())
            
        # 3. Final regression head
        output = model.fc(current_features)
        
    return output.cpu().numpy(), attention_maps


def temporal_binning(A: np.ndarray, bins: int) -> np.ndarray:
    """
    Downsamples attention maps by dividing sequence length into spatial bins
    and averaging values within blocks.
    
    Args:
        A: numpy array of shape (SeqLen, SeqLen), (Batch, SeqLen, SeqLen),
           or (Batch, nhead, SeqLen, SeqLen).
        bins: Target bin count (e.g. 50, 25).
        
    Returns:
        numpy array downsampled to (..., bins, bins) spatial dimensions.
    """
    seq_len = A.shape[-1]
    if seq_len == bins:
        return A
        
    if seq_len % bins != 0:
        raise ValueError(f"Sequence length {seq_len} is not divisible by downsample bins {bins}")
        
    bin_size = seq_len // bins
    
    if A.ndim == 2:
        # (SeqLen, SeqLen) -> (bins, bin_size, bins, bin_size) -> mean over spatial sub-blocks
        return A.reshape(bins, bin_size, bins, bin_size).mean(axis=(1, 3))
    elif A.ndim == 3:
        # (Batch, SeqLen, SeqLen) -> (Batch, bins, bin_size, bins, bin_size)
        return A.reshape(A.shape[0], bins, bin_size, bins, bin_size).mean(axis=(2, 4))
    elif A.ndim == 4:
        # (Batch, nhead, SeqLen, SeqLen) -> (Batch, nhead, bins, bin_size, bins, bin_size)
        return A.reshape(A.shape[0], A.shape[1], bins, bin_size, bins, bin_size).mean(axis=(3, 5))
    else:
        raise ValueError(f"Unsupported attention array dimensions: {A.ndim}")


def calculate_attention_rollout(attention_maps: list, head_idx: str = "mean") -> np.ndarray:
    """
    Computes cumulative Attention Rollout considering residual connections and layer product.
    R_l = \hat{A}_l @ R_{l-1}, where \hat{A}_l = A_l + I (normalized).
    
    Args:
        attention_maps: List of length num_layers. Each item is shape (Batch, nhead, SeqLen, SeqLen).
        head_idx: 'mean' or specific head index.
        
    Returns:
        Rollout matrix of shape (Batch, SeqLen, SeqLen) (Batch, Query, Key).
    """
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
        
    return R


def get_phase_boundaries(gait_phases: dict, seq_len: int) -> dict:
    """
    Maps percentage boundaries to step boundary coordinates.
    """
    boundaries = {}
    for name, (start_pct, end_pct) in gait_phases.items():
        boundaries[name] = (start_pct * seq_len / 100.0, end_pct * seq_len / 100.0)
    return boundaries


def calculate_phase_matrix(A: np.ndarray, seq_len: int, boundaries: dict) -> np.ndarray:
    """
    Computes a compressed (N_phase, N_phase) average attention matrix.
    
    Args:
        A: Attention map of shape (SeqLen, SeqLen) (Query, Key)
        seq_len: sequence length (e.g. 200)
        boundaries: dict containing phase_name -> (start_step, end_step)
        
    Returns:
        matrix: shape (N_phase, N_phase) where row=Key phase, col=Query phase.
    """
    phase_names = list(boundaries.keys())
    num_phases = len(phase_names)
    matrix = np.zeros((num_phases, num_phases))
    
    for j, q_name in enumerate(phase_names):
        q_start, q_end = boundaries[q_name]
        q_indices = np.arange(int(round(q_start)), int(round(q_end)))
        q_indices = q_indices[(q_indices >= 0) & (q_indices < seq_len)]
        
        for i, k_name in enumerate(phase_names):
            k_start, k_end = boundaries[k_name]
            k_indices = np.arange(int(round(k_start)), int(round(k_end)))
            k_indices = k_indices[(k_indices >= 0) & (k_indices < seq_len)]
            
            if len(q_indices) > 0 and len(k_indices) > 0:
                # Slicing the (Query, Key) sub-block and computing the mean.
                sub_matrix = A[np.ix_(q_indices, k_indices)]
                matrix[i, j] = np.mean(sub_matrix)
                
    return matrix


def compute_global_vmax(attention_maps: list) -> float:
    """
    Computes the 99th percentile of all attention weights across layers, heads, and samples.
    """
    all_vals = []
    for m in attention_maps:
        all_vals.append(m.ravel())
    all_vals_flat = np.concatenate(all_vals)
    return float(np.percentile(all_vals_flat, 99))


# ==========================================
# Visualization & Drawing Helpers
# ==========================================

def draw_gait_phase_elements(ax, n_bins: int, boundaries_pct: dict):
    """
    Draws horizontal/vertical dashed lines for gait phase boundaries on a heatmap axis.
    The boundaries are scaled dynamically to the current number of bins.
    
    Args:
        ax: matplotlib axis
        n_bins: number of spatial bins/pixels in the current heatmap (e.g. 200, 50, 25)
        boundaries_pct: dict mapping phase name -> (start_pct, end_pct)
    """
    ends_pct = sorted(list(set([end for _, end in boundaries_pct.values()])))
    for pct in ends_pct:
        if 0.0 < pct < 100.0:
            idx = pct * n_bins / 100.0
            # Vertical line (Query boundary)
            ax.axvline(idx, color='white', linestyle='--', alpha=0.5, linewidth=1.0)
            # Horizontal line (Key boundary)
            ax.axhline(idx, color='white', linestyle='--', alpha=0.5, linewidth=1.0)


def plot_single_heatmap(ax, matrix: np.ndarray, title: str, vmin: float = 0.0, vmax: float = None, 
                        boundaries_pct: dict = None):
    """
    Helper to plot a transposed heatmap (Query on X-axis, Key on Y-axis).
    Row represents Key, Column represents Query.
    """
    # Transpose map: shape (Query, Key) -> (Key, Query)
    matrix_t = matrix.T
    n_bins = matrix_t.shape[0]
    
    sns.heatmap(
        matrix_t,
        cmap="viridis",
        ax=ax,
        cbar=True,
        square=True,
        vmin=vmin,
        vmax=vmax
    )
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel("Query (% Gait Cycle)", fontsize=10)
    ax.set_ylabel("Key (% Gait Cycle)", fontsize=10)
    ax.invert_yaxis()
    
    ticks = [0.0, 0.25 * n_bins, 0.50 * n_bins, 0.75 * n_bins, 1.0 * n_bins]
    labels = ['0', '25', '50', '75', '100']
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    
    if boundaries_pct:
        draw_gait_phase_elements(ax, n_bins, boundaries_pct)


# ==========================================
# Main Load and Run Routine
# ==========================================

def load_data_and_model(exp_dir, fold, device):
    """
    Loads config, models, and arrays from files.
    """
    config_path = os.path.join(exp_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, "r") as f:
        config = json.load(f)
        
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

    # Instantiate model
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
    model_path = os.path.join(exp_dir, f"best_model_fold{fold}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"Loaded model: {model_path}")
    
    # Load data
    inputs = np.load(os.path.join(exp_dir, f"inputs_fold{fold}.npy"))
    preds = np.load(os.path.join(exp_dir, f"preds_fold{fold}.npy"))
    targets = np.load(os.path.join(exp_dir, f"targets_fold{fold}.npy"))
    
    meta_path = os.path.join(exp_dir, f"sample_meta_fold{fold}.csv")
    meta_df = None
    if os.path.exists(meta_path):
        meta_df = pd.read_csv(meta_path)
    
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


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load configuration and dataset arrays
    model, inputs, preds, targets, meta_df, feature_names, target_names = load_data_and_model(
        args.exp_dir, args.fold, device
    )
    
    # Sequence length configurations
    seq_len = inputs.shape[1]
    
    # 2. Setup Gait Phase boundaries
    gait_phases = DEFAULT_GAIT_PHASES
    if args.gait_phases_json:
        try:
            gait_phases = json.loads(args.gait_phases_json)
            print("Using custom gait phase definitions.")
        except Exception as e:
            print(f"Error parsing gait_phases_json: {e}. Using default.")
            
    boundaries = get_phase_boundaries(gait_phases, seq_len)
    
    # 3. Setup Downsample Bin settings
    downsample_ratio = 1.0
    if args.downsample_bins:
        if seq_len % args.downsample_bins != 0:
            print(f"Error: downsample_bins ({args.downsample_bins}) must divide seq_len ({seq_len})")
            sys.exit(1)
        downsample_ratio = seq_len / args.downsample_bins
        print(f"Downsampling active: {seq_len} steps -> {args.downsample_bins} bins (ratio={downsample_ratio})")

    # 4. Extract raw attention maps (Batch, nhead, SeqLen, SeqLen)
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    print("Extracting attention weights (running model forward pass)...")
    _, attention_maps = extract_attention_maps(model, inputs_tensor, device)
    
    # 5. Compute global attention parameters (vmax)
    global_vmax = compute_global_vmax(attention_maps)
    print(f"Computed Global Attention vmax (99th percentile): {global_vmax:.5f}")
    vmax = global_vmax if args.shared_cmap else None
    
    # 6. Initialize output subdirectories
    output_base = args.output_dir if args.output_dir else os.path.join(args.exp_dir, "attention_plots", f"fold{args.fold}")
    subdirs = ["layerwise", "headwise", "rollout", "phase_matrix"]
    for sd in subdirs:
        os.makedirs(os.path.join(output_base, sd), exist_ok=True)
        
    # 7. Apply filtering by subject and/or condition
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
                print(f"Selected sample_idx {args.sample_idx} to match filter criteria.")

    # ==========================================
    # Run Mode executions
    # ==========================================
    
    if args.mode == "single":
        s_idx = args.sample_idx
        if s_idx >= len(inputs) or s_idx < 0:
            print(f"Error: sample_idx {s_idx} is out of bounds (0 to {len(inputs) - 1})")
            sys.exit(1)
            
        sub_name = meta_df.loc[s_idx, 'subject_name'] if meta_df is not None else f"sub{s_idx}"
        cond_name = meta_df.loc[s_idx, 'condition_name'] if meta_df is not None else "cond"
        
        # ------------------------------------------
        # A. Process Maps for Sample
        # ------------------------------------------
        num_layers = len(attention_maps)
        
        # Extract individual maps for layer/head/rollout
        layer_averages = []
        for l_idx in range(num_layers):
            # shape: (nhead, SeqLen, SeqLen)
            s_map = attention_maps[l_idx][s_idx]
            
            # Layer average (mean over head)
            l_avg = np.mean(s_map, axis=0) # (SeqLen, SeqLen)
            
            # Apply binning if requested
            if args.downsample_bins:
                l_avg = temporal_binning(l_avg, args.downsample_bins)
            layer_averages.append(l_avg)
            
            # Save layerwise average individual PNG
            fig_l, ax_l = plt.subplots(figsize=(6, 5))
            plot_single_heatmap(
                ax_l, l_avg, f"Layer {l_idx+1} Mean (Sample {s_idx})",
                vmin=0.0, vmax=vmax, boundaries_pct=gait_phases
            )
            fig_l.savefig(
                os.path.join(output_base, "layerwise", f"layer{l_idx+1}_sample{s_idx}.png"),
                dpi=300, bbox_inches='tight'
            )
            plt.close(fig_l)
            
            # Save headwise individual PNGs
            nheads = s_map.shape[0]
            for h_idx in range(nheads):
                h_map = s_map[h_idx]
                if args.downsample_bins:
                    h_map = temporal_binning(h_map, args.downsample_bins)
                    
                fig_h, ax_h = plt.subplots(figsize=(6, 5))
                plot_single_heatmap(
                    ax_h, h_map, f"Layer {l_idx+1} Head {h_idx} (Sample {s_idx})",
                    vmin=0.0, vmax=vmax, boundaries_pct=gait_phases
                )
                fig_h.savefig(
                    os.path.join(output_base, "headwise", f"layer{l_idx+1}_head{h_idx}_sample{s_idx}.png"),
                    dpi=300, bbox_inches='tight'
                )
                plt.close(fig_h)

        # ------------------------------------------
        # B. Phase-to-Phase Matrix Calculation
        # ------------------------------------------
        # Calculate Phase Matrix based on the mean across layers
        mean_all_layers = np.mean(np.stack([np.mean(attention_maps[l][s_idx], axis=0) for l in range(num_layers)]), axis=0)
        p_matrix = calculate_phase_matrix(mean_all_layers, seq_len, boundaries)
        
        # Save Phase Matrix heatmap
        fig_pm, ax_pm = plt.subplots(figsize=(7, 6))
        sns.heatmap(p_matrix.T, cmap="viridis", annot=True, fmt=".4f", ax=ax_pm, square=True)
        ax_pm.set_title(f"Phase-to-Phase Attention Matrix (Sample {s_idx})", fontsize=12, fontweight='bold')
        ax_pm.set_xlabel("Query (Current Phase)", fontsize=10)
        ax_pm.set_ylabel("Key (Attended Phase)", fontsize=10)
        ax_pm.invert_yaxis()
        fig_pm.savefig(
            os.path.join(output_base, "phase_matrix", f"phase_matrix_sample{s_idx}.png"),
            dpi=300, bbox_inches='tight'
        )
        plt.close(fig_pm)

        # ------------------------------------------
        # C. Attention Rollout Calculation
        # ------------------------------------------
        rollout_batch = calculate_attention_rollout(attention_maps, args.head_idx) # (Batch, SeqLen, SeqLen)
        s_rollout = rollout_batch[s_idx] # (SeqLen, SeqLen)
        
        if args.downsample_bins:
            s_rollout = temporal_binning(s_rollout, args.downsample_bins)
            
        # Save Rollout Heatmap
        fig_r, ax_r = plt.subplots(figsize=(6, 5))
        plot_single_heatmap(
            ax_r, s_rollout, f"Attention Rollout (Sample {s_idx}, Head: {args.head_idx})",
            vmin=0.0, vmax=vmax, boundaries_pct=gait_phases
        )
        fig_r.savefig(
            os.path.join(output_base, "rollout", f"rollout_sample{s_idx}.png"),
            dpi=300, bbox_inches='tight'
        )
        plt.close(fig_r)
        
        # Save Rollout Profile (Key-wise Profile only)
        transposed_rollout = s_rollout.T # (Key, Query)
        key_profile = np.mean(transposed_rollout, axis=1) # Mean over Query
        gait_cycle_pct = np.linspace(0, 100, len(key_profile))
        
        fig_prof, ax_prof = plt.subplots(figsize=(10, 4))
        ax_prof.plot(gait_cycle_pct, key_profile, 'b-', linewidth=2.0, label="Key-wise Profile (Importance as Reference)")
        ax_prof.set_title("Rollout Attention Profile Over Time", fontsize=12, fontweight='bold')
        ax_prof.set_xlabel("Key (% Gait Cycle)", fontsize=10)
        ax_prof.set_ylabel("Attention Weight", fontsize=10)
        ax_prof.grid(True, linestyle=":", alpha=0.6)
        ax_prof.legend()
        ax_prof.set_xlim(0, 100)
        ax_prof.set_xticks([0, 25, 50, 75, 100])
        fig_prof.savefig(
            os.path.join(output_base, "rollout", f"rollout_profile_sample{s_idx}.png"),
            dpi=300, bbox_inches='tight'
        )
        plt.close(fig_prof)

        # Save Attention Profile per Layer (Single mode standalone)
        fig_l_prof, ax_l_prof = plt.subplots(figsize=(10, 4))
        colors = ["tab:blue", "tab:orange", "tab:green"]
        for l_idx in range(num_layers):
            l_profile = np.mean(layer_averages[l_idx], axis=0) # Mean over Query
            gait_cycle_pct_l = np.linspace(0, 100, len(l_profile))
            ax_l_prof.plot(
                gait_cycle_pct_l, l_profile,
                label=f"Layer {l_idx + 1}",
                color=colors[l_idx % len(colors)],
                linewidth=2.0
            )
        ax_l_prof.set_title("Attention Profile per Layer", fontsize=12, fontweight='bold')
        ax_l_prof.set_xlabel("Key (% Gait Cycle)", fontsize=10)
        ax_l_prof.set_ylabel("Key-wise Attention Mean", fontsize=10)
        ax_l_prof.grid(True, linestyle=":", alpha=0.6)
        ax_l_prof.legend(loc="upper right")
        ax_l_prof.set_xlim(0, 100)
        ax_l_prof.set_xticks([0, 25, 50, 75, 100])
        fig_l_prof.savefig(
            os.path.join(output_base, "rollout", f"layer_attention_profile_sample{s_idx}.png"),
            dpi=300, bbox_inches='tight'
        )
        plt.close(fig_l_prof)



    elif args.mode == "aggregate":
        # ------------------------------------------
        # Aggregate Analysis Mode
        # ------------------------------------------
        agg_inputs = inputs[sample_indices]
        agg_preds = preds[sample_indices]
        agg_targets = targets[sample_indices]
        agg_maps = [m[sample_indices] for m in attention_maps]
        
        num_samples = len(agg_inputs)
        num_layers = len(agg_maps)
        
        # 1. Calculate and save Layerwise Aggregated Maps, and Aggregate Head-wise Maps
        layer_averages = []
        for l_idx in range(num_layers):
            # Mean across batch and heads
            agg_m = np.mean(np.mean(agg_maps[l_idx], axis=1), axis=0) # (SeqLen, SeqLen)
            if args.downsample_bins:
                agg_m = temporal_binning(agg_m, args.downsample_bins)
            layer_averages.append(agg_m)
            
            # Save aggregated layerwise heatmap individual PNG
            fig_l, ax_l = plt.subplots(figsize=(6, 5))
            plot_single_heatmap(
                ax_l, agg_m, f"Aggregated Layer {l_idx+1} Mean (N={num_samples})",
                vmin=0.0, vmax=vmax, boundaries_pct=gait_phases
            )
            fig_l.savefig(
                os.path.join(output_base, "layerwise", f"layer{l_idx+1}_mean_aggregate.png"),
                dpi=300, bbox_inches='tight'
            )
            plt.close(fig_l)
            
            # Save aggregate head-wise mean heatmaps (layer x head mean aggregate)
            # shape of agg_maps[l_idx]: (Batch, nhead, SeqLen, SeqLen)
            nheads = agg_maps[l_idx].shape[1]
            for h_idx in range(nheads):
                head_map = np.mean(agg_maps[l_idx][:, h_idx], axis=0) # (SeqLen, SeqLen)
                if args.downsample_bins:
                    head_map = temporal_binning(head_map, args.downsample_bins)
                    
                fig_h, ax_h = plt.subplots(figsize=(6, 5))
                plot_single_heatmap(
                    ax_h, head_map, f"Aggregated Layer {l_idx+1} Head {h_idx} Mean (N={num_samples})",
                    vmin=0.0, vmax=vmax, boundaries_pct=gait_phases
                )
                fig_h.savefig(
                    os.path.join(output_base, "headwise", f"layer{l_idx+1}_head{h_idx}_mean_aggregate.png"),
                    dpi=300, bbox_inches='tight'
                )
                plt.close(fig_h)

        # 2. Aggregated Phase Matrix
        mean_all_layers_agg = np.mean(np.stack([np.mean(np.mean(agg_maps[l], axis=1), axis=0) for l in range(num_layers)]), axis=0)
        p_matrix_agg = calculate_phase_matrix(mean_all_layers_agg, seq_len, boundaries)
        
        fig_pm, ax_pm = plt.subplots(figsize=(7, 6))
        sns.heatmap(p_matrix_agg.T, cmap="viridis", annot=True, fmt=".4f", ax=ax_pm, square=True)
        ax_pm.set_title(f"Aggregated Phase Matrix (N={num_samples})", fontsize=12, fontweight='bold')
        ax_pm.set_xlabel("Query (Current Phase)", fontsize=10)
        ax_pm.set_ylabel("Key (Attended Phase)", fontsize=10)
        ax_pm.invert_yaxis()
        fig_pm.savefig(
            os.path.join(output_base, "phase_matrix", f"phase_matrix_aggregate.png"),
            dpi=300, bbox_inches='tight'
        )
        plt.close(fig_pm)

        # 3. Aggregated Attention Rollout
        rollout_batch = calculate_attention_rollout(agg_maps, args.head_idx) # (Batch, SeqLen, SeqLen)
        agg_rollout = np.mean(rollout_batch, axis=0) # (SeqLen, SeqLen)
        
        if args.downsample_bins:
            agg_rollout = temporal_binning(agg_rollout, args.downsample_bins)
            
        # Save rollout heatmap
        fig_r, ax_r = plt.subplots(figsize=(6, 5))
        plot_single_heatmap(
            ax_r, agg_rollout, f"Aggregated Rollout (N={num_samples}, Head: {args.head_idx})",
            vmin=0.0, vmax=vmax, boundaries_pct=gait_phases
        )
        
        filter_suffix = ""
        if args.subject_name:
            filter_suffix += f"_{args.subject_name}"
        if args.condition_name:
            filter_suffix += f"_{args.condition_name}"
            
        fig_r.savefig(
            os.path.join(output_base, "rollout", f"rollout_aggregate{filter_suffix}.png"),
            dpi=300, bbox_inches='tight'
        )
        plt.close(fig_r)
        
        # Save rollout profile aggregate (Key-wise mean rollout profile)
        transposed_rollout = np.transpose(rollout_batch, (0, 2, 1)) # (Batch, Key, Query)
        batch_key_profiles = np.mean(transposed_rollout, axis=2) # (Batch, Key)
        
        if args.downsample_bins:
            bin_size = seq_len // args.downsample_bins
            batch_key_profiles = batch_key_profiles.reshape(num_samples, args.downsample_bins, bin_size).mean(axis=2)
            
        mean_prof = np.mean(batch_key_profiles, axis=0)
        std_prof = np.std(batch_key_profiles, axis=0)
        gait_cycle_pct = np.linspace(0, 100, len(mean_prof))
        
        fig_prof, ax_prof = plt.subplots(figsize=(10, 4))
        ax_prof.plot(gait_cycle_pct, mean_prof, color="tab:blue", linewidth=2.0, label="Global Mean Rollout Profile")
        ax_prof.fill_between(
            gait_cycle_pct,
            np.maximum(0, mean_prof - std_prof),
            mean_prof + std_prof,
            color="tab:blue",
            alpha=0.2
        )
        ax_prof.set_title("Global Attention Rollout Profile", fontsize=12, fontweight='bold')
        ax_prof.set_xlabel("Key (% Gait Cycle)", fontsize=10)
        ax_prof.set_ylabel("Rollout Weight", fontsize=10)
        ax_prof.grid(True, linestyle=":", alpha=0.6)
        ax_prof.legend()
        ax_prof.set_xlim(0, 100)
        ax_prof.set_xticks([0, 25, 50, 75, 100])
        fig_prof.savefig(
            os.path.join(output_base, "rollout", f"rollout_profile_aggregate{filter_suffix}.png"),
            dpi=300, bbox_inches='tight'
        )
        plt.close(fig_prof)

        # Save Attention Profile per Layer (Aggregate mode standalone)
        fig_layers, ax_layers = plt.subplots(figsize=(10, 4))
        colors = ["tab:blue", "tab:orange", "tab:green"]
        for l_idx in range(num_layers):
            h_mean = np.mean(agg_maps[l_idx], axis=1) # (Batch, SeqLen, SeqLen)
            profiles = np.mean(h_mean, axis=1) # (Batch, SeqLen)
            
            if args.downsample_bins:
                bin_size = seq_len // args.downsample_bins
                profiles = profiles.reshape(num_samples, args.downsample_bins, bin_size).mean(axis=2)
                
            mean_prof_l = np.mean(profiles, axis=0)
            std_prof_l = np.std(profiles, axis=0)
            gait_cycle_pct_l = np.linspace(0, 100, len(mean_prof_l))
            
            ax_layers.plot(
                gait_cycle_pct_l, mean_prof_l,
                label=f"Layer {l_idx + 1}",
                color=colors[l_idx % len(colors)],
                linewidth=2.0
            )
            ax_layers.fill_between(
                gait_cycle_pct_l,
                np.maximum(0, mean_prof_l - std_prof_l),
                mean_prof_l + std_prof_l,
                color=colors[l_idx % len(colors)],
                alpha=0.1
            )
            
        ax_layers.set_title("Aggregated Attention Profile per Layer (Mean ± Std)", fontsize=12, fontweight='bold')
        ax_layers.set_xlabel("Key (% Gait Cycle)", fontsize=10)
        ax_layers.set_ylabel("Key-wise Attention Mean", fontsize=10)
        ax_layers.grid(True, linestyle=":", alpha=0.6)
        ax_layers.legend(loc="upper right")
        ax_layers.set_xlim(0, 100)
        ax_layers.set_xticks([0, 25, 50, 75, 100])
        fig_layers.savefig(
            os.path.join(output_base, "rollout", "layer_attention_profile_aggregate.png"),
            dpi=300, bbox_inches='tight'
        )
        plt.close(fig_layers)



    # 8. Interactive Show Check
    if not args.no_show:
        try:
            plt.show()
        except Exception:
            print("No graphic display detected or backend not available. Plots were saved to file.")


if __name__ == "__main__":
    main()
