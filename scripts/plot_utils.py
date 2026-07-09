import numpy as np

def get_gait_phase_ticks(stride_type: str, num_steps: int = 200):
    """
    Returns time mapping and ticks for a specific gait stride pattern.
    
    Args:
        stride_type (str): The pattern name (e.g., '0.5', '1.0_post_swing', '1.5').
        num_steps (int): The number of sequence steps.
        
    Returns:
        pct_array (np.ndarray): The continuous percentage mapping (len = num_steps).
        tick_indices (list): Pixel/step indices for heatmap ticks.
        tick_vals (list): Actual percentage values (e.g. 0, 20, 60).
        tick_labels (list): String labels (e.g. '0', '20', '60').
    """
    if stride_type == '0.5':
        start_pct, end_pct = 0, 60
    elif stride_type in ['1.0_pre_swing', 'pre']:
        start_pct, end_pct = -60, 60
    elif stride_type == '1.5':
        start_pct, end_pct = -60, 100
    else:
        # Default covers '1.0', '1.0_post_swing', 'post'
        start_pct, end_pct = 0, 100

    pct_array = np.linspace(start_pct, end_pct, num_steps)
    
    # 20% intervals
    min_tick = int(np.ceil(start_pct / 20.0)) * 20
    max_tick = int(np.floor(end_pct / 20.0)) * 20
    
    tick_vals = list(range(min_tick, max_tick + 1, 20))
    
    # Ensure 0% and 60% are always included if they fall within bounds
    for important in [0, 60]:
        if start_pct <= important <= end_pct and important not in tick_vals:
            tick_vals.append(important)
            
    tick_vals = sorted(list(set(tick_vals)))
    
    tick_indices = []
    tick_labels = []
    for val in tick_vals:
        idx = (val - start_pct) / (end_pct - start_pct) * (num_steps - 1)
        tick_indices.append(idx)
        tick_labels.append(str(val))
        
    return pct_array, tick_indices, tick_vals, tick_labels
