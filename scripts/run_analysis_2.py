import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.append(repo_root)

BASE_DIR = os.path.join(repo_root, "outputs/experiments")

EXPERIMENT_DIRS = {
    "Transformer": os.path.join(BASE_DIR, "transformer_X_0.5_Y_0.5_transformer_20260707_052451"),
    "GCN": os.path.join(BASE_DIR, "hybrid_grf_X_0.5_Y_0.5_hybrid_grf_20260803_010931"),
    "EdgeConv": os.path.join(BASE_DIR, "hybrid_edge_X_0.5_Y_0.5_hybrid_edge_20260731_070706"),
}

OUTPUT_DIR = os.path.join(repo_root, "outputs/analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_FEATURES = ["Fx", "Fy", "Fz"]

def main():
    print("--- Starting Analysis 2: Gait Cycle Error Comparison ---")
    
    data = {m: {} for m in EXPERIMENT_DIRS}
    
    for model_name, exp_dir in EXPERIMENT_DIRS.items():
        if not os.path.exists(exp_dir):
            print(f"Directory not found: {exp_dir}")
            return
            
        fold_files = glob.glob(os.path.join(exp_dir, "preds_fold*.npy"))
        for fold in range(1, len(fold_files) + 1):
            preds_path = os.path.join(exp_dir, f"preds_fold{fold}.npy")
            targets_path = os.path.join(exp_dir, f"targets_fold{fold}.npy")
            meta_path = os.path.join(exp_dir, f"sample_meta_fold{fold}.csv")
            
            data[model_name][fold] = {
                "preds": np.load(preds_path),       # shape: (N, 200, 3)
                "targets": np.load(targets_path),   # shape: (N, 200, 3)
                "meta": pd.read_csv(meta_path)
            }
            
    baseline_model = "Transformer"
    num_folds = len(data[baseline_model])
    
    # Store aggregated subject MAE per model: {model: {feature: []}}
    # Each list will contain arrays of shape (200,) for each subject
    subject_maes = {m: {f: [] for f in TARGET_FEATURES} for m in EXPERIMENT_DIRS}
    
    for fold in range(1, num_folds + 1):
        meta_base = data[baseline_model][fold]["meta"]
        unique_subjects = meta_base["subject_name"].unique()
        
        for sub in unique_subjects:
            mask_base = (meta_base["subject_name"] == sub).values
            targets = data[baseline_model][fold]["targets"][mask_base]
            
            for model_name in EXPERIMENT_DIRS:
                meta_m = data[model_name][fold]["meta"]
                mask_m = (meta_m["subject_name"] == sub).values
                preds = data[model_name][fold]["preds"][mask_m]
                
                # abs error: shape (N_sub, 200, 3)
                abs_err = np.abs(preds - targets)
                
                # mean over strides (axis 0), resulting in (200, 3)
                mean_abs_err_stride = np.mean(abs_err, axis=0)
                
                for i, feat in enumerate(TARGET_FEATURES):
                    subject_maes[model_name][feat].append(mean_abs_err_stride[:, i])
                    
    # subject_maes[model][feat] is a list of 12 arrays of shape (200,)
    # Convert to shape (12, 200)
    for model_name in EXPERIMENT_DIRS:
        for feat in TARGET_FEATURES:
            subject_maes[model_name][feat] = np.array(subject_maes[model_name][feat])
            
    # Calculate means and stds over subjects
    results_df = pd.DataFrame({"gait_cycle_percent": np.linspace(0, 100, 200)})
    
    for feat in TARGET_FEATURES:
        for model_name in EXPERIMENT_DIRS:
            mean_mae = np.mean(subject_maes[model_name][feat], axis=0)
            std_mae = np.std(subject_maes[model_name][feat], axis=0, ddof=1)
            
            results_df[f"{model_name}_{feat}_MAE_mean"] = mean_mae
            results_df[f"{model_name}_{feat}_MAE_std"] = std_mae
            
            if model_name != "Transformer":
                # Paired difference per subject: (12, 200)
                diff = subject_maes[model_name][feat] - subject_maes["Transformer"][feat]
                results_df[f"{model_name}_minus_Transformer_{feat}_mean"] = np.mean(diff, axis=0)
                results_df[f"{model_name}_minus_Transformer_{feat}_std"] = np.std(diff, axis=0, ddof=1)
                
    results_df.to_csv(os.path.join(OUTPUT_DIR, "gait_cycle_error_comparison.csv"), index=False)
    
    # Plotting
    for feat in TARGET_FEATURES:
        # Plot 1: Mean Absolute Error over Gait Cycle
        plt.figure(figsize=(10, 6))
        for model_name, color in zip(["Transformer", "GCN", "EdgeConv"], ["blue", "green", "red"]):
            mean = results_df[f"{model_name}_{feat}_MAE_mean"]
            std = results_df[f"{model_name}_{feat}_MAE_std"]
            x = results_df["gait_cycle_percent"]
            
            plt.plot(x, mean, label=model_name, color=color)
            plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
            
        plt.title(f"Mean Absolute Error across Gait Cycle ({feat})")
        plt.xlabel("Gait Cycle (%)")
        plt.ylabel("MAE (%BW)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"fig_analysis2_mae_wave_{feat}.png"), dpi=300)
        plt.close()
        
        # Plot 2: Difference vs Transformer
        plt.figure(figsize=(10, 6))
        for model_name, color in zip(["GCN", "EdgeConv"], ["green", "red"]):
            diff_mean = results_df[f"{model_name}_minus_Transformer_{feat}_mean"]
            diff_std = results_df[f"{model_name}_minus_Transformer_{feat}_std"]
            x = results_df["gait_cycle_percent"]
            
            plt.plot(x, diff_mean, label=f"{model_name} - Transformer", color=color)
            plt.fill_between(x, diff_mean - diff_std, diff_mean + diff_std, color=color, alpha=0.2)
            
        plt.axhline(0, color='black', linestyle='--')
        plt.title(f"MAE Difference from Transformer ({feat})")
        plt.xlabel("Gait Cycle (%)")
        plt.ylabel("Delta MAE (Negative means better)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"fig_analysis2_diff_wave_{feat}.png"), dpi=300)
        plt.close()

    print("Analysis 2 completed. Results saved to", OUTPUT_DIR)

if __name__ == "__main__":
    main()
