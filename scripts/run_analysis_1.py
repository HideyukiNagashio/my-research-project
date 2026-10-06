import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

# Add project root to path so we can import src
import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from src.training.metrics import calculate_metrics

BASE_DIR = os.path.join(repo_root, "outputs/experiments")

EXPERIMENT_DIRS = {
    "Transformer": os.path.join(BASE_DIR, "transformer_X_0.5_Y_0.5_transformer_20260707_052451"),
    "GCN": os.path.join(BASE_DIR, "hybrid_grf_X_0.5_Y_0.5_hybrid_grf_20260803_010931"),
    "EdgeConv": os.path.join(BASE_DIR, "hybrid_edge_X_0.5_Y_0.5_hybrid_edge_20260731_070706"),
}

OUTPUT_DIR = os.path.join(repo_root, "outputs/analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_FEATURES = ["Fx", "Fy", "Fz"]

def get_effect_size(x, y):
    # Cohen's d for paired samples
    diff = np.array(x) - np.array(y)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    if std_diff == 0:
        return 0
    return mean_diff / std_diff

def main():
    print("--- Starting Analysis 1: Subject-wise Delta NRMSE ---")
    
    # Store predictions: {model_name: {fold: {preds: ..., targets: ..., meta: ...}}}
    data = {m: {} for m in EXPERIMENT_DIRS}
    
    for model_name, exp_dir in EXPERIMENT_DIRS.items():
        if not os.path.exists(exp_dir):
            print(f"Directory not found: {exp_dir}")
            return
            
        fold_files = glob.glob(os.path.join(exp_dir, "preds_fold*.npy"))
        num_folds = len(fold_files)
        print(f"{model_name}: Found {num_folds} folds")
        
        for fold in range(1, num_folds + 1):
            preds_path = os.path.join(exp_dir, f"preds_fold{fold}.npy")
            targets_path = os.path.join(exp_dir, f"targets_fold{fold}.npy")
            meta_path = os.path.join(exp_dir, f"sample_meta_fold{fold}.csv")
            
            preds = np.load(preds_path)
            targets = np.load(targets_path)
            meta = pd.read_csv(meta_path)
            
            data[model_name][fold] = {
                "preds": preds,
                "targets": targets,
                "meta": meta
            }
            
    # Calculate NRMSE per subject per model
    subject_metrics_list = []
    
    # Assume all models have same folds and same metadata
    baseline_model = "Transformer"
    num_folds = len(data[baseline_model])
    
    for fold in range(1, num_folds + 1):
        meta_base = data[baseline_model][fold]["meta"]
        unique_subjects = meta_base["subject_name"].unique()
        
        for sub in unique_subjects:
            mask_base = (meta_base["subject_name"] == sub).values
            targets = data[baseline_model][fold]["targets"][mask_base]
            
            row = {"subject": sub, "fold": fold}
            for model_name in EXPERIMENT_DIRS:
                meta_m = data[model_name][fold]["meta"]
                mask_m = (meta_m["subject_name"] == sub).values
                preds = data[model_name][fold]["preds"][mask_m]
                
                # Verify targets match roughly to ensure data alignment
                targets_m = data[model_name][fold]["targets"][mask_m]
                assert np.allclose(targets, targets_m, atol=1e-3), f"Targets mismatch for {sub} in {model_name}"
                
                metrics = calculate_metrics(targets, preds)
                per_feat = metrics['per_feature']
                
                for i, feat in enumerate(TARGET_FEATURES):
                    row[f"{model_name}_{feat}_NRMSE"] = per_feat[i]['nrmse']
                    
            subject_metrics_list.append(row)
            
    df_metrics = pd.DataFrame(subject_metrics_list)
    df_metrics.to_csv(os.path.join(OUTPUT_DIR, "subject_wise_nrmse.csv"), index=False)
    
    # Delta NRMSE
    df_delta = df_metrics.copy()
    for feat in TARGET_FEATURES:
        for model in ["GCN", "EdgeConv"]:
            df_delta[f"{model}_Delta_{feat}"] = df_delta[f"{model}_{feat}_NRMSE"] - df_delta[f"Transformer_{feat}_NRMSE"]
            
    cols_to_keep = ["subject"] + [f"{model}_Delta_{feat}" for feat in TARGET_FEATURES for model in ["GCN", "EdgeConv"]]
    df_delta_out = df_delta[cols_to_keep]
    df_delta_out.to_csv(os.path.join(OUTPUT_DIR, "subject_wise_delta_nrmse.csv"), index=False)
    
    # Statistics
    print("\n--- Statistics for Fx ---")
    for model in ["GCN", "EdgeConv"]:
        col = f"{model}_Delta_Fx"
        deltas = df_delta[col].values
        n_improved = np.sum(deltas < 0)
        
        print(f"Model: {model}")
        print(f"  Improved subjects: {n_improved} / {len(deltas)}")
        print(f"  Mean Delta NRMSE: {np.mean(deltas):.4f}")
        print(f"  Median Delta NRMSE: {np.median(deltas):.4f}")
        print(f"  Std Delta NRMSE: {np.std(deltas, ddof=1):.4f}")
        print(f"  Min Delta NRMSE: {np.min(deltas):.4f}")
        print(f"  Max Delta NRMSE: {np.max(deltas):.4f}")
        
        # Paired test
        x = df_metrics[f"{model}_Fx_NRMSE"].values
        y = df_metrics["Transformer_Fx_NRMSE"].values
        stat, p = wilcoxon(x, y)
        d = get_effect_size(x, y)
        print(f"  Wilcoxon p-value: {p:.4f}")
        print(f"  Cohen's d: {d:.4f}\n")
        
    # Plotting Fx Delta NRMSE
    plt.figure(figsize=(10, 6))
    plot_df = pd.melt(df_delta, id_vars=["subject"], value_vars=["GCN_Delta_Fx", "EdgeConv_Delta_Fx"], 
                      var_name="Model", value_name="Delta_NRMSE")
    plot_df["Model"] = plot_df["Model"].apply(lambda x: x.split('_')[0])
    
    sns.barplot(data=plot_df, x="subject", y="Delta_NRMSE", hue="Model")
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Delta NRMSE for Fx per Subject (Baseline = Transformer)")
    plt.ylabel("Delta NRMSE (Negative is better)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_analysis1_delta_nrmse_bar.png"), dpi=300)
    plt.close()
    
    # Paired plot
    plt.figure(figsize=(8, 6))
    sns.pointplot(data=df_metrics, x="subject", y="Transformer_Fx_NRMSE", color="blue", label="Transformer", markers="o")
    sns.pointplot(data=df_metrics, x="subject", y="EdgeConv_Fx_NRMSE", color="red", label="EdgeConv", markers="x")
    plt.title("Paired Plot: Transformer vs EdgeConv (Fx NRMSE)")
    plt.ylabel("NRMSE")
    plt.xticks(rotation=45)
    plt.legend(["Transformer", "EdgeConv"])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_analysis1_paired_plot.png"), dpi=300)
    plt.close()

    print("Analysis 1 completed. Results saved to", OUTPUT_DIR)

if __name__ == "__main__":
    main()
