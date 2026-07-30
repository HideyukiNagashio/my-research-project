import re

with open("scripts/visualize_attention.py", "r") as f:
    content = f.read()

# 1. Update get_phase_boundaries
old1 = """def get_phase_boundaries(gait_phases: dict, seq_len: int) -> dict:
    \"\"\"
    Maps percentage boundaries to step boundary coordinates.
    \"\"\"
    boundaries = {}
    for name, (start_pct, end_pct) in gait_phases.items():
        boundaries[name] = (start_pct * seq_len / 100.0, end_pct * seq_len / 100.0)
    return boundaries"""

new1 = """def get_phase_boundaries(gait_phases: dict, seq_len: int, start_pct: float = 0.0, end_pct: float = 100.0) -> dict:
    \"\"\"
    Maps percentage boundaries to step boundary coordinates dynamically based on sequence bounds.
    \"\"\"
    boundaries = {}
    for name, (p_start, p_end) in gait_phases.items():
        idx_start = (p_start - start_pct) / (end_pct - start_pct) * (seq_len - 1)
        idx_end = (p_end - start_pct) / (end_pct - start_pct) * (seq_len - 1)
        boundaries[name] = (idx_start, idx_end)
    return boundaries"""
content = content.replace(old1, new1)


# 2. Update calculate_phase_matrix
old2 = """def calculate_phase_matrix(A: np.ndarray, seq_len: int, boundaries: dict) -> np.ndarray:
    \"\"\"
    Computes a compressed (N_phase, N_phase) average attention matrix.
    
    Args:
        A: Attention map of shape (SeqLen, SeqLen) (Query, Key)
        seq_len: sequence length (e.g. 200)
        boundaries: dict containing phase_name -> (start_step, end_step)
        
    Returns:
        matrix: shape (N_phase, N_phase) where row=Key phase, col=Query phase.
    \"\"\"
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
            k_indices = k_indices[(k_indices >= 0) & (k_indices < seq_len)]"""

new2 = """def calculate_phase_matrix(A: np.ndarray, seq_len: int, boundaries_q: dict, boundaries_k: dict) -> np.ndarray:
    \"\"\"
    Computes a compressed (N_phase, N_phase) average attention matrix.
    
    Args:
        A: Attention map of shape (SeqLen, SeqLen) (Query, Key)
        seq_len: sequence length (e.g. 200)
        boundaries_q: dict containing phase_name -> (start_step, end_step) for Query (Output)
        boundaries_k: dict containing phase_name -> (start_step, end_step) for Key (Input)
        
    Returns:
        matrix: shape (N_phase, N_phase) where row=Key phase, col=Query phase.
    \"\"\"
    phase_names = list(boundaries_q.keys())
    num_phases = len(phase_names)
    matrix = np.zeros((num_phases, num_phases))
    
    for j, q_name in enumerate(phase_names):
        q_start, q_end = boundaries_q[q_name]
        q_indices = np.arange(int(round(q_start)), int(round(q_end)))
        q_indices = q_indices[(q_indices >= 0) & (q_indices < seq_len)]
        
        for i, k_name in enumerate(phase_names):
            k_start, k_end = boundaries_k[k_name]
            k_indices = np.arange(int(round(k_start)), int(round(k_end)))
            k_indices = k_indices[(k_indices >= 0) & (k_indices < seq_len)]"""
content = content.replace(old2, new2)


# 3. Update draw_gait_phase_elements and plot_single_heatmap
old3 = """def draw_gait_phase_elements(ax, n_bins: int, boundaries_pct: dict, start_pct: float, end_pct: float):
    \"\"\"
    Draws horizontal/vertical dashed lines for gait phase boundaries on a heatmap axis.
    The boundaries are scaled dynamically to the current number of bins.
    
    Args:
        ax: matplotlib axis
        n_bins: number of spatial bins/pixels in the current heatmap (e.g. 200, 50, 25)
        boundaries_pct: dict mapping phase name -> (start_pct, end_pct)
        start_pct: the start percentage of the axis (e.g., -60, 0)
        end_pct: the end percentage of the axis (e.g., 60, 100)
    \"\"\"
    ends_pct = sorted(list(set([end for _, end in boundaries_pct.values()])))
    for pct in ends_pct:
        if start_pct < pct < end_pct:
            idx = (pct - start_pct) / (end_pct - start_pct) * (n_bins - 1)
            # Vertical line (Query boundary)
            ax.axvline(idx, color='white', linestyle='--', alpha=0.5, linewidth=1.0)
            # Horizontal line (Key boundary)
            ax.axhline(idx, color='white', linestyle='--', alpha=0.5, linewidth=1.0)


def plot_single_heatmap(ax, matrix: np.ndarray, title: str, vmin: float = 0.0, vmax: float = None, 
                        boundaries_pct: dict = None, tick_indices: list = None, tick_labels: list = None,
                        start_pct: float = 0.0, end_pct: float = 100.0):
    \"\"\"
    Helper to plot a transposed heatmap (Query on X-axis, Key on Y-axis).
    Row represents Key, Column represents Query.
    \"\"\"
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
    
    if tick_indices is None:
        tick_indices = [0.0, 0.25 * n_bins, 0.50 * n_bins, 0.75 * n_bins, 1.0 * n_bins]
        tick_labels = ['0', '25', '50', '75', '100']
        
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_indices)
    ax.set_yticklabels(tick_labels)
    
    if boundaries_pct:
        draw_gait_phase_elements(ax, n_bins, boundaries_pct, start_pct, end_pct)"""

new3 = """def draw_gait_phase_elements(ax, n_bins: int, boundaries_pct: dict, start_pct_x: float, end_pct_x: float, start_pct_y: float, end_pct_y: float):
    \"\"\"
    Draws horizontal/vertical dashed lines for gait phase boundaries on a heatmap axis.
    The boundaries are scaled dynamically to the current number of bins.
    \"\"\"
    ends_pct = sorted(list(set([end for _, end in boundaries_pct.values()])))
    for pct in ends_pct:
        if start_pct_x < pct < end_pct_x:
            idx_x = (pct - start_pct_x) / (end_pct_x - start_pct_x) * (n_bins - 1)
            # Vertical line (Query boundary, X-axis)
            ax.axvline(idx_x, color='white', linestyle='--', alpha=0.5, linewidth=1.0)
        if start_pct_y < pct < end_pct_y:
            idx_y = (pct - start_pct_y) / (end_pct_y - start_pct_y) * (n_bins - 1)
            # Horizontal line (Key boundary, Y-axis)
            ax.axhline(idx_y, color='white', linestyle='--', alpha=0.5, linewidth=1.0)


def plot_single_heatmap(ax, matrix: np.ndarray, title: str, vmin: float = 0.0, vmax: float = None, 
                        boundaries_pct: dict = None, 
                        tick_indices_x: list = None, tick_labels_x: list = None,
                        tick_indices_y: list = None, tick_labels_y: list = None,
                        start_pct_x: float = 0.0, end_pct_x: float = 100.0,
                        start_pct_y: float = 0.0, end_pct_y: float = 100.0):
    \"\"\"
    Helper to plot a transposed heatmap (Query on X-axis, Key on Y-axis).
    Row represents Key, Column represents Query.
    \"\"\"
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
    
    if tick_indices_x is None:
        tick_indices_x = [0.0, 0.25 * n_bins, 0.50 * n_bins, 0.75 * n_bins, 1.0 * n_bins]
        tick_labels_x = ['0', '25', '50', '75', '100']
    if tick_indices_y is None:
        tick_indices_y = tick_indices_x
        tick_labels_y = tick_labels_x
        
    ax.set_xticks(tick_indices_x)
    ax.set_xticklabels(tick_labels_x)
    ax.set_yticks(tick_indices_y)
    ax.set_yticklabels(tick_labels_y)
    
    if boundaries_pct:
        draw_gait_phase_elements(ax, n_bins, boundaries_pct, start_pct_x, end_pct_x, start_pct_y, end_pct_y)"""
content = content.replace(old3, new3)


# 4. Update main() variable loading
old4 = """    # Sequence length configurations
    seq_len = inputs.shape[1]
    
    # Get dynamic gait cycle mapping for the input sequence (Self-Attention)
    pct_array, tick_indices, tick_vals, tick_labels = get_gait_phase_ticks(stride_type_X, seq_len)
    start_pct, end_pct = pct_array[0], pct_array[-1]
    
    # 2. Setup Gait Phase boundaries
    gait_phases = DEFAULT_GAIT_PHASES
    if args.gait_phases_json:
        try:
            gait_phases = json.loads(args.gait_phases_json)
            print("Using custom gait phase definitions.")
        except Exception as e:
            print(f"Error parsing gait_phases_json: {e}. Using default.")
            
    boundaries = get_phase_boundaries(gait_phases, seq_len)"""

new4 = """    # Sequence length configurations
    seq_len = inputs.shape[1]
    
    # Get dynamic gait cycle mapping for Input and Output sequences
    pct_array_in, tick_indices_in, tick_vals_in, tick_labels_in = get_gait_phase_ticks(stride_type_X, seq_len)
    start_pct_in, end_pct_in = pct_array_in[0], pct_array_in[-1]
    
    pct_array_out, tick_indices_out, tick_vals_out, tick_labels_out = get_gait_phase_ticks(stride_type_Y, seq_len)
    start_pct_out, end_pct_out = pct_array_out[0], pct_array_out[-1]
    
    # 2. Setup Gait Phase boundaries
    gait_phases = DEFAULT_GAIT_PHASES
    if args.gait_phases_json:
        try:
            gait_phases = json.loads(args.gait_phases_json)
            print("Using custom gait phase definitions.")
        except Exception as e:
            print(f"Error parsing gait_phases_json: {e}. Using default.")
            
    boundaries_in = get_phase_boundaries(gait_phases, seq_len, start_pct_in, end_pct_in)
    boundaries_out = get_phase_boundaries(gait_phases, seq_len, start_pct_out, end_pct_out)"""
content = content.replace(old4, new4)


# 5. Fix plot_single_heatmap usages in main
content = content.replace("""                vmin=0.0, vmax=vmax, boundaries_pct=gait_phases,
                tick_indices=tick_indices, tick_labels=tick_labels, start_pct=start_pct, end_pct=end_pct""",
                """                vmin=0.0, vmax=vmax, boundaries_pct=gait_phases,
                tick_indices_x=tick_indices_in, tick_labels_x=tick_labels_in,
                tick_indices_y=tick_indices_in, tick_labels_y=tick_labels_in,
                start_pct_x=start_pct_in, end_pct_x=end_pct_in,
                start_pct_y=start_pct_in, end_pct_y=end_pct_in""")

# Only replace the Rollout usages where tick_indices_x needs to be tick_indices_out
content = content.replace("""plot_single_heatmap(
            ax_r, s_rollout, f"Attention Rollout (Sample {s_idx}, Head: {args.head_idx})",
            vmin=0.0, vmax=vmax, boundaries_pct=gait_phases,
            tick_indices_x=tick_indices_in, tick_labels_x=tick_labels_in,
            tick_indices_y=tick_indices_in, tick_labels_y=tick_labels_in,
            start_pct_x=start_pct_in, end_pct_x=end_pct_in,
            start_pct_y=start_pct_in, end_pct_y=end_pct_in
        )""", """plot_single_heatmap(
            ax_r, s_rollout, f"Attention Rollout (Sample {s_idx}, Head: {args.head_idx})",
            vmin=0.0, vmax=vmax, boundaries_pct=gait_phases,
            tick_indices_x=tick_indices_out, tick_labels_x=tick_labels_out,
            tick_indices_y=tick_indices_in, tick_labels_y=tick_labels_in,
            start_pct_x=start_pct_out, end_pct_x=end_pct_out,
            start_pct_y=start_pct_in, end_pct_y=end_pct_in
        )""")
content = content.replace("""plot_single_heatmap(
            ax_r, agg_rollout, f"Aggregated Rollout (N={num_samples}, Head: {args.head_idx})",
            vmin=0.0, vmax=vmax, boundaries_pct=gait_phases,
            tick_indices_x=tick_indices_in, tick_labels_x=tick_labels_in,
            tick_indices_y=tick_indices_in, tick_labels_y=tick_labels_in,
            start_pct_x=start_pct_in, end_pct_x=end_pct_in,
            start_pct_y=start_pct_in, end_pct_y=end_pct_in
        )""", """plot_single_heatmap(
            ax_r, agg_rollout, f"Aggregated Rollout (N={num_samples}, Head: {args.head_idx})",
            vmin=0.0, vmax=vmax, boundaries_pct=gait_phases,
            tick_indices_x=tick_indices_out, tick_labels_x=tick_labels_out,
            tick_indices_y=tick_indices_in, tick_labels_y=tick_labels_in,
            start_pct_x=start_pct_out, end_pct_x=end_pct_out,
            start_pct_y=start_pct_in, end_pct_y=end_pct_in
        )""")


# Fix Phase Matrix calls
content = content.replace("calculate_phase_matrix(s_rollout_raw, seq_len, boundaries)", "calculate_phase_matrix(s_rollout_raw, seq_len, boundaries_q=boundaries_out, boundaries_k=boundaries_in)")
content = content.replace("calculate_phase_matrix(agg_rollout_raw, seq_len, boundaries)", "calculate_phase_matrix(agg_rollout_raw, seq_len, boundaries_q=boundaries_out, boundaries_k=boundaries_in)")

# For aggregate head/layer maps where it didn't have tick_indices passed explicitly in plot_single_heatmap:
old_agg_h = """plot_single_heatmap(
                    ax_h, head_map, f"Aggregated Layer {l_idx+1} Head {h_idx} Mean (N={num_samples})",
                    vmin=0.0, vmax=vmax, boundaries_pct=gait_phases
                )"""
new_agg_h = """plot_single_heatmap(
                    ax_h, head_map, f"Aggregated Layer {l_idx+1} Head {h_idx} Mean (N={num_samples})",
                    vmin=0.0, vmax=vmax, boundaries_pct=gait_phases,
                    tick_indices_x=tick_indices_in, tick_labels_x=tick_labels_in,
                    tick_indices_y=tick_indices_in, tick_labels_y=tick_labels_in,
                    start_pct_x=start_pct_in, end_pct_x=end_pct_in,
                    start_pct_y=start_pct_in, end_pct_y=end_pct_in
                )"""
content = content.replace(old_agg_h, new_agg_h)

old_agg_l = """plot_single_heatmap(
                ax_l, agg_m, f"Aggregated Layer {l_idx+1} Mean (N={num_samples})",
                vmin=0.0, vmax=vmax, boundaries_pct=gait_phases
            )"""
new_agg_l = """plot_single_heatmap(
                ax_l, agg_m, f"Aggregated Layer {l_idx+1} Mean (N={num_samples})",
                vmin=0.0, vmax=vmax, boundaries_pct=gait_phases,
                tick_indices_x=tick_indices_in, tick_labels_x=tick_labels_in,
                tick_indices_y=tick_indices_in, tick_labels_y=tick_labels_in,
                start_pct_x=start_pct_in, end_pct_x=end_pct_in,
                start_pct_y=start_pct_in, end_pct_y=end_pct_in
            )"""
content = content.replace(old_agg_l, new_agg_l)

old_agg_r = """plot_single_heatmap(
            ax_r, agg_rollout, f"Aggregated Rollout (N={num_samples}, Head: {args.head_idx})",
            vmin=0.0, vmax=vmax, boundaries_pct=gait_phases
        )"""
new_agg_r = """plot_single_heatmap(
            ax_r, agg_rollout, f"Aggregated Rollout (N={num_samples}, Head: {args.head_idx})",
            vmin=0.0, vmax=vmax, boundaries_pct=gait_phases,
            tick_indices_x=tick_indices_out, tick_labels_x=tick_labels_out,
            tick_indices_y=tick_indices_in, tick_labels_y=tick_labels_in,
            start_pct_x=start_pct_out, end_pct_x=end_pct_out,
            start_pct_y=start_pct_in, end_pct_y=end_pct_in
        )"""
content = content.replace(old_agg_r, new_agg_r)


# Replace gait_cycle_pct, tick_vals, tick_labels for profiles
content = content.replace("gait_cycle_pct = pct_array", "gait_cycle_pct = pct_array_in")
content = content.replace("gait_cycle_pct_l = pct_array", "gait_cycle_pct_l = pct_array_in")

content = content.replace("ax_prof.set_xlim(0, 100)", "ax_prof.set_xlim(start_pct_in, end_pct_in)")
content = content.replace("ax_prof.set_xlim(start_pct, end_pct)", "ax_prof.set_xlim(start_pct_in, end_pct_in)")
content = content.replace("ax_prof.set_xticks([0, 25, 50, 75, 100])", "ax_prof.set_xticks(tick_vals_in)\\n        ax_prof.set_xticklabels(tick_labels_in)")

content = content.replace("ax_prof.set_xticks(tick_vals)", "ax_prof.set_xticks(tick_vals_in)")
content = content.replace("ax_prof.set_xticklabels(tick_labels)", "ax_prof.set_xticklabels(tick_labels_in)")

content = content.replace("ax_l_prof.set_xlim(start_pct, end_pct)", "ax_l_prof.set_xlim(start_pct_in, end_pct_in)")
content = content.replace("ax_l_prof.set_xticks(tick_vals)", "ax_l_prof.set_xticks(tick_vals_in)")
content = content.replace("ax_l_prof.set_xticklabels(tick_labels)", "ax_l_prof.set_xticklabels(tick_labels_in)")

content = content.replace("ax_layers.set_xlim(start_pct, end_pct)", "ax_layers.set_xlim(start_pct_in, end_pct_in)")
content = content.replace("ax_layers.set_xticks(tick_vals)", "ax_layers.set_xticks(tick_vals_in)")
content = content.replace("ax_layers.set_xticklabels(tick_labels)", "ax_layers.set_xticklabels(tick_labels_in)")

with open("scripts/visualize_attention.py", "w") as f:
    f.write(content)
