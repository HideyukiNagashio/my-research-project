import re

with open("scripts/visualize_gradient.py", "r") as f:
    content = f.read()

# 1. Add plot_feature_importance_bars function
plot_func = """
def plot_feature_importance_bars(abs_sums, rel_pcts, feature_names, out_label, save_path=None, timers=None):
    \"\"\"Plots and saves a horizontal bar chart of absolute sum and relative % feature importance.\"\"\"
    if timers is not None:
        t_plot_start = time.time()
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    
    y_pos = np.arange(len(feature_names))
    
    # Left subplot: Absolute Sum
    ax1.barh(y_pos, abs_sums, color='skyblue')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(feature_names)
    ax1.invert_yaxis()  # Labels read top-to-bottom
    ax1.set_xlabel('Absolute Gradient Sum')
    ax1.set_title(f"Absolute Importance ({out_label})")
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Right subplot: Relative Percentage
    ax2.barh(y_pos, rel_pcts, color='salmon')
    ax2.set_xlabel('Relative Importance (%)')
    ax2.set_title(f"Relative Importance % ({out_label})")
    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add percentage labels on the bars
    for i, v in enumerate(rel_pcts):
        ax2.text(v + 0.5, i, f"{v:.1f}%", va='center', fontsize=9)
        
    plt.suptitle(f"Feature Importance Map ({out_label})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if timers is not None:
        timers['Plotting'] += time.time() - t_plot_start
        
    if save_path:
        if timers is not None:
            t_save_start = time.time()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        if timers is not None:
            timers['Save Figure'] += time.time() - t_save_start
            
    plt.close()


# =====================================================================
# Model and Data Loader
"""
content = content.replace("# =====================================================================\n# Model and Data Loader", plot_func)

# 2. Add folder creation in production mode
content = content.replace('os.makedirs(os.path.join(fold_out_dir, "phase"), exist_ok=True)', 
                          'os.makedirs(os.path.join(fold_out_dir, "phase"), exist_ok=True)\n        os.makedirs(os.path.join(fold_out_dir, "importance"), exist_ok=True)')

# 3. Add Demo mode computation
demo_approach3 = """        print(f"\\nDemo run completed successfully. View plots in: {demo_dir}")"""
demo_approach4 = """        # Approach 4: Feature Importance
        print("\\n--- Approach 4: Computing Feature Importance (Demo) ---")
        for o_c in demo_out_cols:
            all_dynamics = compute_dynamics_map_all_features(model, input_data, out_col=o_c)
            abs_sums = np.sum(all_dynamics, axis=(1, 2))
            total_sum = np.sum(abs_sums)
            rel_pcts = (abs_sums / total_sum) * 100.0 if total_sum > 0 else abs_sums
            
            save_path = os.path.join(fold_out_dir, "importance", f"feature_importance_{target_names[o_c]}.png")
            plot_feature_importance_bars(abs_sums, rel_pcts, feature_names, target_names[o_c], save_path)
            print(f"Saved: {save_path}")
            
        print(f"\\nDemo run completed successfully. View plots in: {demo_dir}")"""
content = content.replace(demo_approach3, demo_approach4)

# 4. Add Production mode computation
prod_approach3_end = """            t_app3_total = time.time() - t_app3_start
            print(f"Completed Phase-wise Smoothed Maps for Output {out_label}")
            print(f"Approach 3 (Plotting & Saving) Time: {t_app3_total:.3f}s")
            print(f"  Plotting    : {timers_app3['Plotting']:.3f}s")
            print(f"  Save Figure : {timers_app3['Save Figure']:.3f}s")"""

prod_approach4 = """            t_app3_total = time.time() - t_app3_start
            print(f"Completed Phase-wise Smoothed Maps for Output {out_label}")
            print(f"Approach 3 (Plotting & Saving) Time: {t_app3_total:.3f}s")
            print(f"  Plotting    : {timers_app3['Plotting']:.3f}s")
            print(f"  Save Figure : {timers_app3['Save Figure']:.3f}s")
            
            # --- 5. Approach 4: Feature Importance Bars ---
            print(f"\\n--> Running Approach 4: Feature Importance for Output {out_label}...")
            t_app4_start = time.time()
            timers_app4 = {'Plotting': 0.0, 'Save Figure': 0.0}
            
            abs_sums = np.sum(mean_3d_dynamics, axis=(1, 2))
            total_sum = np.sum(abs_sums)
            rel_pcts = (abs_sums / total_sum) * 100.0 if total_sum > 0 else abs_sums
            
            save_path = os.path.join(
                fold_out_dir, 
                "importance", 
                f"feature_importance_{out_label}{sample_suffix}.png"
            )
            plot_feature_importance_bars(abs_sums, rel_pcts, feature_names, out_label, save_path, timers=timers_app4)
            t_app4_total = time.time() - t_app4_start
            print(f"Completed Feature Importance Maps for Output {out_label}")
            print(f"Approach 4 (Plotting & Saving) Time: {t_app4_total:.3f}s")
            print(f"  Plotting    : {timers_app4['Plotting']:.3f}s")
            print(f"  Save Figure : {timers_app4['Save Figure']:.3f}s")"""
content = content.replace(prod_approach3_end, prod_approach4)

with open("scripts/visualize_gradient.py", "w") as f:
    f.write(content)
