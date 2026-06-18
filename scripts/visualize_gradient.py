#!/usr/bin/env python3
"""
Gait Analysis Explainability Framework (Input-Gradient / Vanilla Backpropagation)
Extracts and visualizes input feature and time step contributions (sensitivities)
to output predictions across defined gait phases.
Supports standalone demo mode and batch-processing of actual trained experiments.
"""

import os
import sys
import glob
import json
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib
# Set non-interactive backend if no display is available
if 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import time

# Add the project root to the python path to import src modules if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from src.models import get_model
except ImportError:
    # Fallback if executing outside the repository environment
    get_model = None

# Default gait phase ratio (in % of stride cycle)
DEFAULT_GAIT_PHASES = {
    'LR':  (0.0, 10.0),   # Loading Response (0 - 20 step)
    'MSt': (10.0, 30.0),  # Mid Stance (20 - 60 step)
    'TSt': (30.0, 50.0),  # Terminal Stance (60 - 100 step)
    'PSw': (50.0, 60.0),  # Pre Swing (100 - 120 step)
    'ISw': (60.0, 75.0),  # Initial Swing (120 - 150 step)
    'MSw': (75.0, 85.0),  # Mid Swing (150 - 170 step)
    'TSw': (85.0, 100.0)  # Terminal Swing (170 - 200 step)
}

def get_phase_indices(seq_len=200):
    """Convert percentage phase bounds to absolute step index slices."""
    indices = {}
    for phase_name, (start_pct, end_pct) in DEFAULT_GAIT_PHASES.items():
        start_idx = int(start_pct * seq_len / 100.0)
        end_idx = int(end_pct * seq_len / 100.0)
        indices[phase_name] = (start_idx, end_idx)
    return indices


# =====================================================================
# Core Computation Algorithms
# =====================================================================

def compute_dynamics_map_slow(model, input_data, out_col, in_col):
    """
    Original slow implementation doing 200 forwards and 200 backwards.
    Used exclusively for verification comparisons.
    """
    model.eval()
    seq_len = input_data.shape[1]
    dynamics_map = np.zeros((seq_len, seq_len))
    
    for x in range(seq_len):
        if input_data.grad is not None:
            input_data.grad.zero_()
        model.zero_grad()
        
        outputs = model(input_data)
        score = outputs[0, x, out_col]
        score.backward(retain_graph=True)
        
        if input_data.grad is not None:
            grad_slice = input_data.grad[0, :, in_col].cpu().numpy()
            dynamics_map[:, x] = np.abs(grad_slice)
            
    return dynamics_map


def compute_dynamics_map(model, input_data, out_col, in_col):
    """
    Optimized implementation with 1 forward and 200 backwards.
    Maintains exact numerical consistency with compute_dynamics_map_slow.
    """
    t_start = time.time()
    
    model.eval()
    seq_len = input_data.shape[1]
    dynamics_map = np.zeros((seq_len, seq_len))
    
    # 1. Forward pass only once
    t_forward_start = time.time()
    outputs = model(input_data)  # Shape: (Batch, seq_len, out_dim)
    t_forward = time.time() - t_forward_start
    
    t_backward = 0.0
    for x in range(seq_len):
        # 2. Reset gradients (model.zero_grad is preserved per user specification)
        if input_data.grad is not None:
            input_data.grad.zero_()
        model.zero_grad()
        
        score = outputs[0, x, out_col]
        
        # 3. Backward pass. Retain graph for all steps except the last one to release memory.
        is_last = (x == seq_len - 1)
        t_backward_start = time.time()
        score.backward(retain_graph=not is_last)
        t_backward += time.time() - t_backward_start
        
        # 4. Extract gradients
        if input_data.grad is not None:
            grad_slice = input_data.grad[0, :, in_col].cpu().numpy()
            dynamics_map[:, x] = np.abs(grad_slice)
            
    t_total = time.time() - t_start
    t_other = t_total - t_forward - t_backward
    
    print(f"\n[Profile: compute_dynamics_map]")
    print(f"  Forward total:    {t_forward:.4f}s ({t_forward/t_total*100:.1f}%)")
    print(f"  Backward total:   {t_backward:.4f}s ({t_backward/t_total*100:.1f}%)")
    print(f"  Other processing: {t_other:.4f}s ({t_other/t_total*100:.1f}%)")
    print(f"  Total time:       {t_total:.4f}s\n")
    
    return dynamics_map


def compute_overall_average_map(model, input_data, out_col):
    """
    Computes an overall average sensitivity map of shape [14, 200]
    across the entire output sequence.
    
    Args:
        model: PyTorch model
        input_data: Input tensor of shape (1, SeqLen, InputDim) with requires_grad=True
        out_col: Target output index (0~2)
        
    Returns:
        mean_map: numpy array of shape (InputDim, SeqLen) (vertical: input_column, horizontal: input_time)
    """
    model.eval()
    seq_len = input_data.shape[1]
    in_dim = input_data.shape[2]
    accumulated_grads = np.zeros((seq_len, in_dim))
    
    for x in range(seq_len):
        if input_data.grad is not None:
            input_data.grad.zero_()
        model.zero_grad()
        
        outputs = model(input_data)
        score = outputs[0, x, out_col]
        score.backward(retain_graph=True)
        
        if input_data.grad is not None:
            accumulated_grads += np.abs(input_data.grad[0].cpu().numpy())
            
    # Simple average over output time dimension
    mean_grads = accumulated_grads / seq_len
    # Transpose to obtain shape [InputDim, SeqLen] (vertical: features, horizontal: time)
    return mean_grads.T


def compute_phase_smoothed_maps(model, input_data, out_col):
    """
    Computes sensitivity maps [14, 200] smoothed (averaged) over 7 gait phases.
    
    Args:
        model: PyTorch model
        input_data: Input tensor of shape (1, SeqLen, InputDim) with requires_grad=True
        out_col: Target output index (0~2)
        
    Returns:
        phase_maps: dict mapping phase name to a numpy array of shape (InputDim, SeqLen)
    """
    model.eval()
    seq_len = input_data.shape[1]
    in_dim = input_data.shape[2]
    phase_slices = get_phase_indices(seq_len)
    phase_maps = {}
    
    for phase_name, (start_idx, end_idx) in phase_slices.items():
        phase_steps = end_idx - start_idx
        if phase_steps == 0:
            phase_maps[phase_name] = np.zeros((in_dim, seq_len))
            continue
            
        accumulated_grads = np.zeros((seq_len, in_dim))
        
        for x in range(start_idx, end_idx):
            if input_data.grad is not None:
                input_data.grad.zero_()
            model.zero_grad()
            
            outputs = model(input_data)
            score = outputs[0, x, out_col]
            score.backward(retain_graph=True)
            
            if input_data.grad is not None:
                accumulated_grads += np.abs(input_data.grad[0].cpu().numpy())
                
        # Average inside the phase duration
        mean_grads = accumulated_grads / phase_steps
        # Shape [InputDim, SeqLen]
        phase_maps[phase_name] = mean_grads.T
        
    return phase_maps


# =====================================================================
# Plotting Helpers
# =====================================================================

def plot_dynamics_map(dynamics_map, out_label, in_label, save_path=None):
    """Plots and saves the dynamics heatmap."""
    plt.figure(figsize=(10, 8))
    
    # Generate labels for 0% to 100% of gait cycle (every 10% / 20 steps)
    time_labels = ["" for _ in range(200)]
    for i in range(0, 200, 20):
        time_labels[i] = f"{int(i * 0.5)}%"
    time_labels[-1] = "100%"
    
    sns.heatmap(dynamics_map, cmap="rocket_r", xticklabels=time_labels, yticklabels=time_labels, cbar_kws={'label': 'Gradient Magnitude'})
    plt.title(f"Dynamics Map ({out_label} Output vs {in_label} Input)\nVertical: Input Time $y$, Horizontal: Output Time $x$")
    plt.xlabel("Output Time $x$ (% of Gait Cycle)")
    plt.ylabel("Input Time $y$ (% of Gait Cycle)")
    plt.gca().invert_yaxis()  # Keep 0 at bottom
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_overall_average_map(mean_map, out_label, feature_names, save_path=None):
    """Plots and saves the overall average sensitivity heatmap."""
    plt.figure(figsize=(12, 6))
    
    # Generate labels for 0% to 100% of gait cycle (every 10% / 20 steps)
    time_labels = ["" for _ in range(200)]
    for i in range(0, 200, 20):
        time_labels[i] = f"{int(i * 0.5)}%"
    time_labels[-1] = "100%"
    
    sns.heatmap(mean_map, cmap="rocket_r", xticklabels=time_labels, yticklabels=feature_names, cbar_kws={'label': 'Mean Gradient Magnitude'})
    plt.title(f"Overall Average Sensitivity Map ({out_label} Output)\nVertical: Input Columns, Horizontal: Input Time $y$")
    plt.xlabel("Input Time $y$ (% of Gait Cycle)")
    plt.ylabel("Input Features")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_phase_smoothed_maps(phase_maps, out_label, feature_names, save_path=None):
    """Plots and saves the 7 phase subplots in a single figure."""
    fig, axes = plt.subplots(7, 1, figsize=(12, 22), sharex=True)
    phases = list(phase_maps.keys())
    
    # Determine a common max value for color scale mapping consistency
    global_max = max(p_map.max() for p_map in phase_maps.values()) if len(phase_maps) > 0 else 1.0
    if global_max <= 0:
        global_max = 1.0
        
    # Generate labels for 0% to 100% of gait cycle (every 10% / 20 steps)
    time_labels = ["" for _ in range(200)]
    for i in range(0, 200, 20):
        time_labels[i] = f"{int(i * 0.5)}%"
    time_labels[-1] = "100%"
    
    for i, phase_name in enumerate(phases):
        ax = axes[i]
        sns.heatmap(
            phase_maps[phase_name], 
            cmap="rocket_r", 
            xticklabels=time_labels, 
            yticklabels=feature_names if i == 0 or i == 3 or i == 6 else False,
            vmin=0.0, 
            vmax=global_max, 
            ax=ax,
            cbar_kws={'label': 'Gradient'} if i == 3 else None
        )
        ax.set_title(f"Phase: {phase_name} ({DEFAULT_GAIT_PHASES[phase_name][0]:.0f}% - {DEFAULT_GAIT_PHASES[phase_name][1]:.0f}%)")
        ax.set_ylabel("Input Features" if i == 0 or i == 3 or i == 6 else "")
        
    plt.xlabel("Input Time $y$ (% of Gait Cycle)")
    plt.suptitle(f"Phase-wise Smoothed Sensitivity Maps ({out_label} Output)\nVertical: Features, Horizontal: Input Time", y=0.99, fontsize=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()


# =====================================================================
# Model and Data Loader
# =====================================================================

def load_data_and_model(exp_dir, fold, device):
    """Loads configuration, model structure, weights, and test arrays."""
    config_path = os.path.join(exp_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, "r") as f:
        config = json.load(f)
        
    input_type = config.get("input_type", "bilateral")
    target_type = config.get("target_type", "all")
    model_type = config.get("model_type", "transformer")
    
    # Calculate input dimension
    if input_type == 'single_leg': in_dim = 14
    elif input_type == 'bilateral': in_dim = 28
    elif input_type == 'pressure_single': in_dim = 8
    elif input_type == 'pressure_bilateral': in_dim = 16
    elif input_type == 'imu_single': in_dim = 6
    elif input_type == 'imu_bilateral': in_dim = 12
    else: raise ValueError(f"Unknown input_type {input_type}")

    # Calculate output dimension
    if target_type == 'all': out_dim = 12
    elif target_type == 'angles_only': out_dim = 9
    elif target_type == 'angles_6dof': out_dim = 6
    elif target_type == 'angles_3dof': out_dim = 3
    elif target_type == 'grf_only': out_dim = 3
    else: raise ValueError(f"Unknown target_type {target_type}")

    # Reconstruct model kwargs
    model_kwargs = {
        'input_dim': in_dim,
        'output_dim': out_dim,
        'dropout_prob': config.get("dropout", 0.1)
    }
    if model_type == 'cnn':
        model_kwargs.update({
            'hidden_dim': config.get("hidden_dim", 256),
            'kernel_size': config.get("kernel_size", 5)
        })
    elif model_type == 'bilstm':
        model_kwargs.update({
            'hidden_dim': config.get("hidden_dim", 256),
            'num_layers': config.get("num_layers", 3)
        })
    elif model_type == 'transformer':
        model_kwargs.update({
            'd_model': config.get("d_model", 128),
            'nhead': config.get("nhead", 4),
            'num_layers': config.get("num_layers", 3),
            'dim_feedforward': config.get("dim_feedforward", 256)
        })
        
    if get_model is None:
        raise ImportError("Cannot import get_model. Make sure scripts are run from the project root.")
        
    model = get_model(model_type, **model_kwargs)
    
    # Load model weights
    model_path = os.path.join(exp_dir, f"best_model_fold{fold}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print(f"Loaded {model_type.upper()} model from: {model_path}")
    
    # Load test data array
    inputs_path = os.path.join(exp_dir, f"inputs_fold{fold}.npy")
    if not os.path.exists(inputs_path):
        raise FileNotFoundError(f"Inputs array file not found: {inputs_path}")
    inputs = np.load(inputs_path)
    
    # Load metadata Names if available
    feature_names = []
    feature_path = os.path.join(exp_dir, "feature_names.json")
    if os.path.exists(feature_path):
        with open(feature_path, "r") as f:
            feature_names = json.load(f)
    else:
        feature_names = [f"Input_Col_{i}" for i in range(in_dim)]
        
    target_names = []
    target_path = os.path.join(exp_dir, "target_names.json")
    if os.path.exists(target_path):
        with open(target_path, "r") as f:
            target_names = json.load(f)
    else:
        target_names = [f"Output_Col_{i}" for i in range(out_dim)]
        
    return model, inputs, feature_names, target_names, in_dim, out_dim


# =====================================================================
# Standalone Execution / Testing Routine with Mock Model
# =====================================================================

class MockGaitModel(nn.Module):
    """
    A lightweight mockup model implementing Conv1D and Linear operations
    to simulate structural, temporal, and feature dependencies on gait analysis.
    This generates natural gradient flows (e.g., temporal diagonal bands) for verification.
    """
    def __init__(self, in_dim=14, out_dim=3):
        super().__init__()
        # 1D Convolution over time dimension to blend neighboring steps
        self.conv = nn.Conv1d(
            in_channels=in_dim, 
            out_channels=in_dim, 
            kernel_size=7, 
            padding=3, 
            bias=False
        )
        self.fc = nn.Linear(in_dim, out_dim)
        
        # Initialize with synthetic weights to make the gradient maps structured
        with torch.no_grad():
            self.conv.weight.fill_(0.0)
            for i in range(in_dim):
                self.conv.weight[i, i, 3] = 0.6  # center
                self.conv.weight[i, i, 2] = 0.2  # left
                self.conv.weight[i, i, 4] = 0.2  # right
            
            self.fc.weight.fill_(0.01)
            self.fc.weight[0, 0:8] = 0.5   # Fx -> Foot pressure
            self.fc.weight[1, 8:11] = 0.8  # Fy -> Accelerometer
            self.fc.weight[2, 11:14] = 0.9 # Fz -> Gyroscope

    def forward(self, x):
        x_t = x.transpose(1, 2)
        x_conv = self.conv(x_t)
        x_out = x_conv.transpose(1, 2)
        return self.fc(x_out)


def parse_args():
    parser = argparse.ArgumentParser(description="Gait Analysis Input-Gradient Explainability Framework")
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=None,
        help="Path to experiment directory containing config.json and fold outputs (optional, triggers demo mode if omitted)"
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="all",
        help="Fold index to visualize (1-6, or 'all' to batch process all folds dynamically)"
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=-1,
        help="Index of test sample to analyze. Set to -1 to average over all samples in the test split."
    )
    parser.add_argument(
        "--out_col",
        type=str,
        default="all",
        help="Output feature index to process (0, 1, 2, or 'all' to process all outputs dynamically)"
    )
    parser.add_argument(
        "--in_col",
        type=str,
        default="all",
        help="Input feature index for Approach 1 dynamics map (or 'all' to process all input features)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # CASE 1: Standalone Demo Mode
    if args.exp_dir is None:
        print("No experiment directory specified. Running in Standalone Demo Mode...")
        demo_dir = "outputs/experiments/explainability_demo"
        os.makedirs(demo_dir, exist_ok=True)
        
        seq_len = 200
        in_dim = 14
        out_dim = 3
        
        model = MockGaitModel(in_dim=in_dim, out_dim=out_dim)
        
        # Create synthetic gait data (1 sample, 200 steps, 14 features)
        t = np.linspace(0, 4 * np.pi, seq_len)
        synthetic_x = np.zeros((1, seq_len, in_dim))
        for i in range(in_dim):
            synthetic_x[0, :, i] = np.sin(t + (i * np.pi / 8)) + np.random.normal(0, 0.05, seq_len)
        input_data = torch.tensor(synthetic_x, dtype=torch.float32, requires_grad=True)
        
        feature_names = [
            "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8",
            "AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"
        ]
        target_names = ["Fx", "Fy", "Fz"]
        
        # Demo targets
        demo_out_cols = [0, 1, 2]
        demo_in_cols = [2, 8, 11]  # P3, AccX, GyroX
        
        # Fold output dir simulation
        fold_out_dir = os.path.join(demo_dir, "fold1")
        os.makedirs(fold_out_dir, exist_ok=True)
        
        # Numerical & Speed Verification for Dynamics Map (Forward 1x vs 200x)
        print("\n--- Running Numerical & Performance Verification (out_col=0, in_col=0) ---")
        
        t0 = time.time()
        dynamics_map_slow = compute_dynamics_map_slow(model, input_data, out_col=0, in_col=0)
        time_slow = time.time() - t0
        print(f"Slow (Forward 200x) Dynamics Map: {time_slow:.4f}s")
        
        t1 = time.time()
        dynamics_map_fast = compute_dynamics_map(model, input_data, out_col=0, in_col=0)
        time_fast = time.time() - t1
        print(f"Fast (Forward 1x) Dynamics Map:  {time_fast:.4f}s")
        
        all_close = np.allclose(dynamics_map_slow, dynamics_map_fast, atol=1e-7, rtol=1e-5)
        max_abs_error = np.max(np.abs(dynamics_map_slow - dynamics_map_fast))
        speedup = time_slow / time_fast if time_fast > 0 else 1.0
        
        print(f"Numerical Equivalence (np.allclose): {all_close}")
        print(f"Max Absolute Error: {max_abs_error:.4e}")
        print(f"Speedup Factor: {speedup:.2f}x")
        print("---------------------------------------------------------------------------------\n")
        
        # Approach 1: Dynamics
        print("\n--- Approach 1: Computing [Input Time y, Output Time x] Dynamics Maps (Demo) ---")
        for o_c in demo_out_cols:
            for i_c in demo_in_cols:
                dynamics_map = compute_dynamics_map(model, input_data, out_col=o_c, in_col=i_c)
                save_path = os.path.join(fold_out_dir, "dynamics", f"dynamics_map_{target_names[o_c]}_vs_{feature_names[i_c]}.png")
                plot_dynamics_map(dynamics_map, target_names[o_c], feature_names[i_c], save_path)
                print(f"Saved: {save_path}")
                
        # Approach 2: Average
        print("\n--- Approach 2: Computing Overall Average Maps (Demo) ---")
        for o_c in demo_out_cols:
            mean_map = compute_overall_average_map(model, input_data, out_col=o_c)
            save_path = os.path.join(fold_out_dir, "average", f"overall_average_map_{target_names[o_c]}.png")
            plot_overall_average_map(mean_map, target_names[o_c], feature_names, save_path)
            print(f"Saved: {save_path}")
            
        # Approach 3: Phase
        print("\n--- Approach 3: Computing Phase-wise Smoothed Maps (Demo) ---")
        for o_c in demo_out_cols:
            phase_maps = compute_phase_smoothed_maps(model, input_data, out_col=o_c)
            save_path = os.path.join(fold_out_dir, "phase", f"phase_wise_smoothed_maps_{target_names[o_c]}.png")
            plot_phase_smoothed_maps(phase_maps, target_names[o_c], feature_names, save_path)
            print(f"Saved: {save_path}")
            
        print(f"\nDemo run completed successfully. View plots in: {demo_dir}")
        return
        
    # CASE 2: Production Experiment Execution Mode
    print(f"Running in Production Mode. Target Directory: {args.exp_dir}")
    
    # Find active folds dynamically
    if args.fold.lower() == "all":
        pattern = os.path.join(args.exp_dir, "inputs_fold*.npy")
        fold_files = sorted(glob.glob(pattern))
        if len(fold_files) == 0:
            raise FileNotFoundError(f"No fold files found matching {pattern}")
        # Extract numbers from inputs_fold{num}.npy
        folds_to_run = []
        for file_path in fold_files:
            basename = os.path.basename(file_path)
            # extracts numbers
            num_str = "".join([c for c in basename if c.isdigit()])
            if num_str:
                folds_to_run.append(int(num_str))
        print(f"Detected folds to process: {folds_to_run}")
    else:
        folds_to_run = [int(args.fold)]
        
    for fold in folds_to_run:
        print(f"\n========================================")
        print(f" Processing Fold {fold} / {len(folds_to_run)}")
        print(f"========================================")
        
        # Load model and data
        model, inputs, feature_names, target_names, in_dim, out_dim = load_data_and_model(
            args.exp_dir, fold, device
        )
        
        # Determine target output columns
        if args.out_col.lower() == "all":
            out_cols = list(range(out_dim))
        else:
            out_cols = [int(args.out_col)]
            
        # Determine target input columns for Approach 1
        if args.in_col.lower() == "all":
            in_cols = list(range(in_dim))
        else:
            in_cols = [int(args.in_col)]
            
        # Select samples to process
        num_samples = inputs.shape[0]
        if args.sample_idx == -1:
            print(f"Averaging gradients over ALL {num_samples} samples in test split...")
            indices_to_avg = list(range(num_samples))
            sample_suffix = "_all_samples_mean"
        else:
            if args.sample_idx < 0 or args.sample_idx >= num_samples:
                raise IndexError(f"sample_idx {args.sample_idx} is out of bounds (0 ~ {num_samples - 1})")
            print(f"Analyzing sample index: {args.sample_idx}")
            indices_to_avg = [args.sample_idx]
            sample_suffix = f"_sample_{args.sample_idx}"
            
        # Create output folders for this fold
        fold_out_dir = os.path.join(args.exp_dir, "gradient_plots", f"fold{fold}")
        os.makedirs(os.path.join(fold_out_dir, "dynamics"), exist_ok=True)
        os.makedirs(os.path.join(fold_out_dir, "average"), exist_ok=True)
        os.makedirs(os.path.join(fold_out_dir, "phase"), exist_ok=True)
        
        seq_len = inputs.shape[1]
        
        # --- Approach 1: Dynamics Maps ---
        print("\n--> Running Approach 1: Dynamics Maps...")
        t0_app1 = time.time()
        for o_c in out_cols:
            out_label = target_names[o_c]
            for i_c in in_cols:
                in_label = feature_names[i_c]
                
                # Accumulated array for averaging
                accum_dynamics = np.zeros((seq_len, seq_len))
                
                for idx in indices_to_avg:
                    # Shape (1, seq_len, in_dim)
                    sample_x = torch.tensor(inputs[idx:idx+1], dtype=torch.float32).to(device)
                    sample_x.requires_grad_(True)
                    # Compute
                    dyn_map = compute_dynamics_map(model, sample_x, out_col=o_c, in_col=i_c)
                    accum_dynamics += dyn_map
                    
                mean_dynamics = accum_dynamics / len(indices_to_avg)
                
                # Plot
                save_path = os.path.join(
                    fold_out_dir, 
                    "dynamics", 
                    f"dynamics_map_{out_label}_vs_{in_label}{sample_suffix}.png"
                )
                plot_dynamics_map(mean_dynamics, out_label, in_label, save_path)
            print(f"Completed Dynamics Maps for Output {out_label}")
        print(f"Approach 1 Time: {time.time()-t0_app1:.3f}s")
            
        # --- Approach 2: Overall Average Maps ---
        print("\n--> Running Approach 2: Overall Average Maps...")
        t0_app2 = time.time()
        for o_c in out_cols:
            out_label = target_names[o_c]
            
            accum_average = np.zeros((in_dim, seq_len))
            
            for idx in indices_to_avg:
                sample_x = torch.tensor(inputs[idx:idx+1], dtype=torch.float32).to(device)
                sample_x.requires_grad_(True)
                mean_map = compute_overall_average_map(model, sample_x, out_col=o_c)
                accum_average += mean_map
                
            mean_average = accum_average / len(indices_to_avg)
            
            save_path = os.path.join(
                fold_out_dir, 
                "average", 
                f"overall_average_map_{out_label}{sample_suffix}.png"
            )
            plot_overall_average_map(mean_average, out_label, feature_names, save_path)
            print(f"Completed Overall Average Map for Output {out_label}")
        print(f"Approach 2 Time: {time.time()-t0_app2:.3f}s")
            
        # --- Approach 3: Phase-wise Smoothed Maps ---
        print("\n--> Running Approach 3: Phase-wise Smoothed Maps...")
        t0_app3 = time.time()
        for o_c in out_cols:
            out_label = target_names[o_c]
            
            # Initialize accumulated phase maps dict
            accum_phase_maps = {}
            for phase_name in DEFAULT_GAIT_PHASES.keys():
                accum_phase_maps[phase_name] = np.zeros((in_dim, seq_len))
                
            for idx in indices_to_avg:
                sample_x = torch.tensor(inputs[idx:idx+1], dtype=torch.float32).to(device)
                sample_x.requires_grad_(True)
                phase_maps = compute_phase_smoothed_maps(model, sample_x, out_col=o_c)
                for phase_name, p_map in phase_maps.items():
                    accum_phase_maps[phase_name] += p_map
                    
            mean_phase_maps = {}
            for phase_name in DEFAULT_GAIT_PHASES.keys():
                mean_phase_maps[phase_name] = accum_phase_maps[phase_name] / len(indices_to_avg)
                
            save_path = os.path.join(
                fold_out_dir, 
                "phase", 
                f"phase_wise_smoothed_maps_{out_label}{sample_suffix}.png"
            )
            plot_phase_smoothed_maps(mean_phase_maps, out_label, feature_names, save_path)
            print(f"Completed Phase-wise Smoothed Maps for Output {out_label}")
        print(f"Approach 3 Time: {time.time()-t0_app3:.3f}s")
            
        print(f"\nFold {fold} processing complete. Outputs saved to: {fold_out_dir}")
        
    print("\nAll tasks completed successfully!")

if __name__ == "__main__":
    main()
