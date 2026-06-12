import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

def holm_correct(p_values):
    """
    Apply Holm-Bonferroni correction to a list of p-values.
    Returns corrected p-values.
    """
    n = len(p_values)
    if n == 0:
        return []
    
    # Sort with indices to keep track of original order
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * n
    
    current_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed_p):
        # Holm formula: p_corr = p * (n - rank)
        p_corr = p * (n - rank)
        p_corr = min(1.0, max(p_corr, current_max))
        current_max = p_corr
        corrected[orig_idx] = p_corr
        
    return corrected

def load_metrics(exp_dir, metric_col):
    summary_path = os.path.join(exp_dir, "subject_summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"subject_summary.csv not found in {exp_dir}. Run calculate_past_subject_metrics.py first.")
    df = pd.read_csv(summary_path)
    if metric_col not in df.columns:
        raise ValueError(f"Metric '{metric_col}' not found in {summary_path}. Available columns: {list(df.columns)}")
    # Sort by subject_name to ensure pairing is aligned
    df = df.sort_values(by="subject_name").reset_index(drop=True)
    return df

def run_wilcoxon(dir1, dir2, label1, label2, metric):
    df1 = load_metrics(dir1, metric)
    df2 = load_metrics(dir2, metric)
    
    # Check subjects matching
    if not df1["subject_name"].equals(df2["subject_name"]):
        raise ValueError("Subject lists do not match between the two experiments.")
        
    subjects = df1["subject_name"].values
    vals1 = df1[metric].values
    vals2 = df2[metric].values
    
    # Run Wilcoxon signed-rank test
    stat, p_val = wilcoxon(vals1, vals2)
    
    print("\n" + "="*60)
    print(f" Wilcoxon Signed-Rank Test ({metric.upper()})")
    print(f" Group 1 ({label1}) vs Group 2 ({label2})")
    print("="*60)
    
    # Table of values
    df_compare = pd.DataFrame({
        "Subject": subjects,
        label1: vals1,
        label2: vals2,
        "Diff (G2 - G1)": vals2 - vals1
    })
    print(df_compare.to_string(index=False))
    print("-"*60)
    print(f"Mean ({label1}): {np.mean(vals1):.4f} ± {np.std(vals1):.4f}")
    print(f"Mean ({label2}): {np.mean(vals2):.4f} ± {np.std(vals2):.4f}")
    print(f"Wilcoxon Statistic: {stat:.1f}")
    print(f"p-value: {p_val:.6f}")
    if p_val < 0.05:
        print("Result: Significant difference (p < 0.05)")
    else:
        print("Result: No significant difference (p >= 0.05)")
    print("="*60 + "\n")

def run_friedman(dirs, labels, metric):
    if len(dirs) < 3:
        raise ValueError("Friedman test requires at least 3 groups.")
    if len(dirs) != len(labels):
        raise ValueError("Number of directories must match the number of labels.")
        
    dfs = []
    for d in dirs:
        dfs.append(load_metrics(d, metric))
        
    # Check subject names match
    subjects = dfs[0]["subject_name"].values
    for idx, df in enumerate(dfs[1:]):
        if not dfs[0]["subject_name"].equals(df["subject_name"]):
            raise ValueError(f"Subject lists do not match between {labels[0]} and {labels[idx+1]}.")
            
    # Prepare data for Friedman: list of arrays, each array is a group
    group_data = [df[metric].values for df in dfs]
    
    # Run Friedman test
    stat, p_val = friedmanchisquare(*group_data)
    
    print("\n" + "="*80)
    print(f" Friedman Test ({metric.upper()})")
    print(" Groups:", ", ".join(labels))
    print("="*80)
    
    df_compare = pd.DataFrame({"Subject": subjects})
    for lbl, vals in zip(labels, group_data):
        df_compare[lbl] = vals
    print(df_compare.to_string(index=False))
    print("-"*80)
    for lbl, vals in zip(labels, group_data):
        print(f"Mean ({lbl}): {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print(f"Friedman Chi-Square Statistic: {stat:.4f}")
    print(f"p-value: {p_val:.6f}")
    
    if p_val < 0.05:
        print("Result: Significant difference exists among groups (p < 0.05). Proceeding to Post-hoc pairwise Wilcoxon tests with Holm correction...")
        print("="*80)
        
        # Post-hoc Pairwise Wilcoxon Tests
        pairs = []
        raw_p_values = []
        num_groups = len(labels)
        
        for i in range(num_groups):
            for j in range(i+1, num_groups):
                lbl1, lbl2 = labels[i], labels[j]
                vals1, vals2 = group_data[i], group_data[j]
                _, p = wilcoxon(vals1, vals2)
                pairs.append((lbl1, lbl2, np.mean(vals1), np.mean(vals2)))
                raw_p_values.append(p)
                
        corrected_p_values = holm_correct(raw_p_values)
        
        print(f"\nPost-hoc Pairwise Comparison (Wilcoxon Signed-Rank with Holm correction):")
        print(f"{'Comparison':<35} | {'Mean 1':<8} | {'Mean 2':<8} | {'Raw p-value':<12} | {'Holm p-value':<12} | {'Significant':<12}")
        print("-" * 97)
        for (lbl1, lbl2, m1, m2), raw_p, corr_p in zip(pairs, raw_p_values, corrected_p_values):
            sig = "Yes" if corr_p < 0.05 else "No"
            comp_name = f"{lbl1} vs {lbl2}"
            print(f"{comp_name:<35} | {m1:<8.4f} | {m2:<8.4f} | {raw_p:<12.6f} | {corr_p:<12.6f} | {sig:<12}")
        print("="*80 + "\n")
    else:
        print("Result: No significant difference exists among groups (p >= 0.05). Post-hoc tests skipped.")
        print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Statistical tests for gait analysis experiment results")
    subparsers = parser.add_subparsers(dest="test_type", help="Type of statistical test")
    
    # Wilcoxon subparser
    wilcoxon_parser = subparsers.add_parser("wilcoxon", help="Wilcoxon signed-rank test for 2 related groups")
    wilcoxon_parser.add_argument("--group1", type=str, required=True, help="Experiment directory 1")
    wilcoxon_parser.add_argument("--group2", type=str, required=True, help="Experiment directory 2")
    wilcoxon_parser.add_argument("--label1", type=str, default="Group 1", help="Label for group 1")
    wilcoxon_parser.add_argument("--label2", type=str, default="Group 2", help="Label for group 2")
    wilcoxon_parser.add_argument("--metric", type=str, default="rmse", help="Metric column to compare (e.g. rmse, mae, r2, Fx_rmse, etc.)")
    
    # Friedman subparser
    friedman_parser = subparsers.add_parser("friedman", help="Friedman test with post-hoc Holm-Wilcoxon for 3+ related groups")
    friedman_parser.add_argument("--dirs", nargs="+", required=True, help="Experiment directories")
    friedman_parser.add_argument("--labels", nargs="+", required=True, help="Labels for each experiment directory")
    friedman_parser.add_argument("--metric", type=str, default="rmse", help="Metric column to compare")
    
    args = parser.parse_args()
    
    if args.test_type == "wilcoxon":
        run_wilcoxon(args.group1, args.group2, args.label1, args.label2, args.metric)
    elif args.test_type == "friedman":
        run_friedman(args.dirs, args.labels, args.metric)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
