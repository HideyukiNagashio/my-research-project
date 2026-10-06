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
    
    global_subjects = []
    
    for fold in range(1, num_folds + 1):
        meta_base = data[baseline_model][fold]["meta"]
        unique_subjects = meta_base["subject_name"].unique()
        
        for sub in unique_subjects:
            global_subjects.append(sub)
            
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
            
    # Calculate phase-based means
    # Assuming the 200 points represent the Stance Phase, which is typically 60% of the Gait Cycle.
    # Therefore, we map Gait Cycle percentage to array index (0-200)
    GAIT_PHASES = {
        'LR (0-10%)':  (0.0, 10.0),   # Loading Response
        'MSt (10-30%)': (10.0, 30.0),  # Mid Stance
        'TSt (30-50%)': (30.0, 50.0),  # Terminal Stance
        'PSw (50-60%)': (50.0, 60.0),  # Pre Swing
    }
    
    phase_results = []
    
    for feat in TARGET_FEATURES:
        for model_name in EXPERIMENT_DIRS:
            # subject_maes[model_name][feat] is (12, 200)
            mae_array = subject_maes[model_name][feat]
            
            for sub_idx, sub_name in enumerate(global_subjects):
                for phase_name, (start_pct, end_pct) in GAIT_PHASES.items():
                    start_idx = int((start_pct / 60.0) * 200)
                    end_idx = int((end_pct / 60.0) * 200)
                    end_idx = min(end_idx, 200) # Safety
                    
                    phase_mae = np.mean(mae_array[sub_idx, start_idx:end_idx])
                    
                    phase_results.append({
                        "Feature": feat,
                        "Model": model_name,
                        "Phase": phase_name,
                        "Subject": sub_name,
                        "MAE": phase_mae
                    })
                    
    df_phases = pd.DataFrame(phase_results)
    df_phases.to_csv(os.path.join(OUTPUT_DIR, "gait_phase_error_comparison.csv"), index=False)
    
    import seaborn as sns
    # Plotting
    for feat in TARGET_FEATURES:
        plt.figure(figsize=(10, 6))
        df_feat = df_phases[df_phases["Feature"] == feat]
        
        sns.barplot(data=df_feat, x="Phase", y="MAE", hue="Model", 
                    palette={"Transformer": "blue", "GCN": "green", "EdgeConv": "red"}, capsize=.05)
        
        plt.title(f"Mean Absolute Error by Gait Phase ({feat})")
        plt.xlabel("Gait Phase")
        plt.ylabel("MAE (%BW)")
        plt.legend(title="Model")
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"fig_analysis2_phase_bar_{feat}.png"), dpi=300)
        plt.close()

    print("Analysis 2 completed. Results saved to", OUTPUT_DIR)

if __name__ == "__main__":
    main()
