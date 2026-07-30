import re

with open("scripts/visualize_gradient.py", "r") as f:
    content = f.read()

# 1. Update get_phase_indices to handle dynamic scaling and filtering
old1 = """def get_phase_indices(seq_len=200):
    \"\"\"Convert percentage phase bounds to absolute step index slices.\"\"\"
    indices = {}
    for phase_name, (start_pct, end_pct) in DEFAULT_GAIT_PHASES.items():
        start_idx = int(start_pct * seq_len / 100.0)
        end_idx = int(end_pct * seq_len / 100.0)
        indices[phase_name] = (start_idx, end_idx)
    return indices"""

new1 = """def get_phase_indices(seq_len=200, seq_start_pct=0.0, seq_end_pct=100.0):
    \"\"\"Convert percentage phase bounds to absolute step index slices.
       Skips phases that fall completely outside the sequence boundaries.\"\"\"
    indices = {}
    for phase_name, (start_pct, end_pct) in DEFAULT_GAIT_PHASES.items():
        # Check if phase overlaps with sequence
        if end_pct <= seq_start_pct or start_pct >= seq_end_pct:
            continue # Phase is outside the sequence

        # Clip phase to sequence bounds
        clamped_start = max(start_pct, seq_start_pct)
        clamped_end = min(end_pct, seq_end_pct)

        # Convert to step indices
        start_idx = int((clamped_start - seq_start_pct) / (seq_end_pct - seq_start_pct) * (seq_len - 1))
        end_idx = int((clamped_end - seq_start_pct) / (seq_end_pct - seq_start_pct) * (seq_len - 1))
        
        indices[phase_name] = (start_idx, end_idx)
    return indices"""
content = content.replace(old1, new1)

# 2. Update plot_phase_smoothed_maps to support dynamic number of subplots
old2 = """def plot_phase_smoothed_maps(phase_maps, out_label, feature_names, tick_indices, tick_labels, x_stride_type, save_path=None, timers=None):
    \"\"\"Plots and saves the 7 phase subplots in a single figure.\"\"\"
    if timers is not None:
        t_plot_start = time.time()
        
    fig, axes = plt.subplots(7, 1, figsize=(12, 22), sharex=True)
    phases = list(phase_maps.keys())
    
    # Determine a common max value for color scale mapping consistency
    global_max = max(p_map.max() for p_map in phase_maps.values()) if len(phase_maps) > 0 else 1.0
    if global_max <= 0:
        global_max = 1.0
        
    for i, phase_name in enumerate(phases):
        ax = axes[i]
        sns.heatmap(
            phase_maps[phase_name], 
            cmap="rocket_r", 
            yticklabels=feature_names if i == 0 or i == 3 or i == 6 else False,
            vmin=0.0, 
            vmax=global_max, 
            ax=ax,
            cbar_kws={'label': 'Gradient'} if i == 3 else None
        )
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels)
        
        ax.set_title(f"Phase: {phase_name} ({DEFAULT_GAIT_PHASES[phase_name][0]:.0f}% - {DEFAULT_GAIT_PHASES[phase_name][1]:.0f}%)")
        ax.set_ylabel("Input Features" if i == 0 or i == 3 or i == 6 else "")"""

new2 = """def plot_phase_smoothed_maps(phase_maps, out_label, feature_names, tick_indices, tick_labels, x_stride_type, save_path=None, timers=None):
    \"\"\"Plots and saves the dynamic phase subplots in a single figure.\"\"\"
    if timers is not None:
        t_plot_start = time.time()
        
    num_phases = len(phase_maps)
    if num_phases == 0:
        return
        
    fig, axes = plt.subplots(num_phases, 1, figsize=(12, 3 * num_phases + 1), sharex=True)
    if num_phases == 1:
        axes = [axes]
    
    phases = list(phase_maps.keys())
    
    # Determine a common max value for color scale mapping consistency
    global_max = max(p_map.max() for p_map in phase_maps.values()) if len(phase_maps) > 0 else 1.0
    if global_max <= 0:
        global_max = 1.0
        
    for i, phase_name in enumerate(phases):
        ax = axes[i]
        show_ylabel = (i == 0 or i == num_phases // 2 or i == num_phases - 1)
        sns.heatmap(
            phase_maps[phase_name], 
            cmap="rocket_r", 
            yticklabels=feature_names if show_ylabel else False,
            vmin=0.0, 
            vmax=global_max, 
            ax=ax,
            cbar_kws={'label': 'Gradient'} if i == num_phases // 2 else None
        )
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels)
        
        ax.set_title(f"Phase: {phase_name} ({DEFAULT_GAIT_PHASES[phase_name][0]:.0f}% - {DEFAULT_GAIT_PHASES[phase_name][1]:.0f}%)")
        ax.set_ylabel("Input Features" if show_ylabel else "")"""
content = content.replace(old2, new2)

# 3. Fix plot_dynamics_map signature and labels
old3 = """def plot_dynamics_map(dynamics_map, out_label, in_label, x_tick_indices, x_tick_labels, y_tick_indices, y_tick_labels, x_stride_type, y_stride_type, save_path=None, timers=None):
    \"\"\"Plots and saves the dynamics heatmap.\"\"\"
    if timers is not None:
        t_plot_start = time.time()
        
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(dynamics_map, cmap="rocket_r", cbar_kws={'label': 'Gradient Magnitude'})
    
    plt.xticks(x_tick_indices, x_tick_labels)
    plt.yticks(y_tick_indices, y_tick_labels)
    
    plt.title(f"Dynamics Map ({out_label} Output vs {in_label} Input)\\nVertical: Input Time (X), Horizontal: Output Time (Y)")
    plt.xlabel(f"Output Time $x$ (% of Gait Cycle: {y_stride_type})")
    plt.ylabel(f"Input Time $y$ (% of Gait Cycle: {x_stride_type})")"""

new3 = """def plot_dynamics_map(dynamics_map, out_label, in_label, x_tick_indices, x_tick_labels, y_tick_indices, y_tick_labels, stride_type_out, stride_type_in, save_path=None, timers=None):
    \"\"\"Plots and saves the dynamics heatmap.\"\"\"
    if timers is not None:
        t_plot_start = time.time()
        
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(dynamics_map, cmap="rocket_r", cbar_kws={'label': 'Gradient Magnitude'})
    
    plt.xticks(x_tick_indices, x_tick_labels)
    plt.yticks(y_tick_indices, y_tick_labels)
    
    plt.title(f"Dynamics Map ({out_label} Output vs {in_label} Input)\\nVertical: Input Time (Y), Horizontal: Output Time (X)")
    plt.xlabel(f"Output Time $x$ (% of Gait Cycle: {stride_type_out})")
    plt.ylabel(f"Input Time $y$ (% of Gait Cycle: {stride_type_in})")"""
content = content.replace(old3, new3)

# 4. Fix usages in Demo Mode
content = content.replace("phase_slices = get_phase_indices(seq_len)", "phase_slices = get_phase_indices(seq_len, 0.0, 100.0)")

content = content.replace("def compute_phase_smoothed_maps(model, input_data, out_col, timers=None):\\n    \"\"\"Wrapper that uses compute_dynamics_map_all_features and slices/averages by phase slices.\"\"\"\\n    all_dynamics = compute_dynamics_map_all_features(model, input_data, out_col, timers=timers)\\n    \\n    if timers is not None:\\n        t_seg_start = time.time()\\n    seq_len = input_data.shape[1]\\n    in_dim = input_data.shape[2]\\n    phase_slices = get_phase_indices(seq_len)",
"def compute_phase_smoothed_maps(model, input_data, out_col, seq_start_pct=0.0, seq_end_pct=100.0, timers=None):\\n    \"\"\"Wrapper that uses compute_dynamics_map_all_features and slices/averages by phase slices.\"\"\"\\n    all_dynamics = compute_dynamics_map_all_features(model, input_data, out_col, timers=timers)\\n    \\n    if timers is not None:\\n        t_seg_start = time.time()\\n    seq_len = input_data.shape[1]\\n    in_dim = input_data.shape[2]\\n    phase_slices = get_phase_indices(seq_len, seq_start_pct, seq_end_pct)")


# 5. Fix usages in Production Mode
old4 = """        _, x_tick_indices, _, x_tick_labels = get_gait_phase_ticks(stride_type_Y, seq_len)
        _, y_tick_indices, _, y_tick_labels = get_gait_phase_ticks(stride_type_X, seq_len)"""

new4 = """        pct_array_out, x_tick_indices, _, x_tick_labels = get_gait_phase_ticks(stride_type_Y, seq_len)
        start_pct_out, end_pct_out = pct_array_out[0], pct_array_out[-1]
        
        pct_array_in, y_tick_indices, _, y_tick_labels = get_gait_phase_ticks(stride_type_X, seq_len)
        start_pct_in, end_pct_in = pct_array_in[0], pct_array_in[-1]"""
content = content.replace(old4, new4)

old5 = "phase_slices = get_phase_indices(seq_len)"
new5 = "phase_slices = get_phase_indices(seq_len, start_pct_out, end_pct_out)"
content = content.replace(old5, new5)

old6 = "plot_dynamics_map(mean_dynamics, out_label, in_label, x_tick_indices, x_tick_labels, y_tick_indices, y_tick_labels, stride_type_X, stride_type_Y, save_path, timers=timers_app1)"
new6 = "plot_dynamics_map(mean_dynamics, out_label, in_label, x_tick_indices, x_tick_labels, y_tick_indices, y_tick_labels, stride_type_out=stride_type_Y, stride_type_in=stride_type_X, save_path=save_path, timers=timers_app1)"
content = content.replace(old6, new6)

with open("scripts/visualize_gradient.py", "w") as f:
    f.write(content)
