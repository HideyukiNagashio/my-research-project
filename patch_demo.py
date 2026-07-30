import re

with open("scripts/visualize_gradient.py", "r") as f:
    content = f.read()

old_demo_init = """        # Demo targets
        demo_out_cols = [0, 1, 2]
        demo_in_cols = [2, 8, 11]  # P3, AccX, GyroX
        
        # Fold output dir simulation"""

new_demo_init = """        # Demo targets
        demo_out_cols = [0, 1, 2]
        demo_in_cols = [2, 8, 11]  # P3, AccX, GyroX
        
        # Mock stride types for demo
        stride_type_X = "1.0_post_swing"
        stride_type_Y = "1.0_post_swing"
        pct_array_out, x_tick_indices, _, x_tick_labels = get_gait_phase_ticks(stride_type_Y, seq_len)
        start_pct_out, end_pct_out = pct_array_out[0], pct_array_out[-1]
        
        pct_array_in, y_tick_indices, _, y_tick_labels = get_gait_phase_ticks(stride_type_X, seq_len)
        start_pct_in, end_pct_in = pct_array_in[0], pct_array_in[-1]
        
        # Fold output dir simulation"""
content = content.replace(old_demo_init, new_demo_init)


old_plot1 = """                plot_dynamics_map(dynamics_map, target_names[o_c], feature_names[i_c], save_path)"""
new_plot1 = """                plot_dynamics_map(dynamics_map, target_names[o_c], feature_names[i_c], x_tick_indices, x_tick_labels, y_tick_indices, y_tick_labels, stride_type_out=stride_type_Y, stride_type_in=stride_type_X, save_path=save_path)"""
content = content.replace(old_plot1, new_plot1)

old_plot2 = """            plot_overall_average_map(mean_map, target_names[o_c], feature_names, save_path)"""
new_plot2 = """            plot_overall_average_map(mean_map, target_names[o_c], feature_names, y_tick_indices, y_tick_labels, stride_type_X, save_path=save_path)"""
content = content.replace(old_plot2, new_plot2)

old_plot3 = """            phase_maps = compute_phase_smoothed_maps(model, input_data, out_col=o_c)
            save_path = os.path.join(fold_out_dir, "phase", f"phase_wise_smoothed_maps_{target_names[o_c]}.png")
            plot_phase_smoothed_maps(phase_maps, target_names[o_c], feature_names, save_path)"""
new_plot3 = """            phase_maps = compute_phase_smoothed_maps(model, input_data, out_col=o_c, seq_start_pct=start_pct_out, seq_end_pct=end_pct_out)
            save_path = os.path.join(fold_out_dir, "phase", f"phase_wise_smoothed_maps_{target_names[o_c]}.png")
            plot_phase_smoothed_maps(phase_maps, target_names[o_c], feature_names, y_tick_indices, y_tick_labels, stride_type_X, save_path=save_path)"""
content = content.replace(old_plot3, new_plot3)


with open("scripts/visualize_gradient.py", "w") as f:
    f.write(content)
