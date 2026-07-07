import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(description='Visualize stride patterns to verify segmentation and GRF shift correction.')
    parser.add_argument('--npz_path', type=str, default='data/processed/normalized/all_data_combined.npz',
                        help='Path to the processed npz file (e.g., all_data_combined.npz or raw_strides/..._raw.npz)')
    parser.add_argument('--sample_idx', type=int, default=10,
                        help='Index of the stride to visualize')
    parser.add_argument('--save_path', type=str, default='grf_shift_check.png',
                        help='Path to save the generated image')
    args = parser.parse_args()

    if not os.path.exists(args.npz_path):
        print(f"Error: File not found -> {args.npz_path}")
        print("Please make sure you have run the preprocessing pipeline first.")
        return

    print(f"Loading data from {args.npz_path} ...")
    data = np.load(args.npz_path)
    
    patterns = ['0.5', '1.0_pre_swing', '1.0_post_swing', '1.5']
    
    # 52ch structure: Fx, Fy, Fz are at indices 23, 24, 25 for the Ipsilateral side.
    ch_indices = [23, 24, 25]
    axes_titles = ['Fx (Anterior-Posterior)', 'Fy (Medial-Lateral)', 'Fz (Vertical)']
    
    fig, axes = plt.subplots(len(patterns), len(ch_indices), figsize=(16, 12), sharey='col')
    fig.suptitle(f'GRF Shift Correction Check (Sample Index: {args.sample_idx})', fontsize=18)
    
    found_patterns = 0

    for i, pat in enumerate(patterns):
        key = f'ensemble_X_{pat}'
        if key not in data:
            print(f"Warning: {key} not found in the npz file.")
            continue
            
        ensemble = data[key]
        if args.sample_idx >= len(ensemble):
            print(f"Error: sample_idx {args.sample_idx} is out of bounds for {key} (Total samples: {len(ensemble)})")
            continue
            
        found_patterns += 1
        
        for j, (title, ch_idx) in enumerate(zip(axes_titles, ch_indices)):
            ax = axes[i, j]
            sig = ensemble[args.sample_idx, :, ch_idx]
            x_axis = np.linspace(0, 100, len(sig))
            
            # 信号のプロット
            ax.plot(x_axis, sig, color='#1f77b4', linewidth=2)
            # 0ラインの描画（遊脚期がここに張り付くべき）
            ax.axhline(0, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label='Zero Line')
            
            if i == 0:
                ax.set_title(title, fontsize=14, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'Pattern: {pat}', fontsize=12, fontweight='bold')
            if i == len(patterns) - 1:
                ax.set_xlabel('Normalized Time / Gait Cycle (%)', fontsize=12)
            
            ax.grid(True, linestyle=':', alpha=0.7)
            if i == 0 and j == 2:
                ax.legend()
                
    if found_patterns == 0:
        print("No patterns could be plotted. Check the file contents or sample index.")
        return

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(args.save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization successfully saved to: {args.save_path}")

if __name__ == '__main__':
    main()
