import pandas as pd

def load_participant_data(participant, condition, mass):
    print(f"\n{'='*55}")
    print(f"  {participant} - {condition.upper()}  (mass: {mass} kg)")
    print(f"{'='*55}")
    df_left  = pd.read_csv(f"data/raw/wearable/{participant}_{condition}_left_foot_data.csv",  header=0)
    df_right = pd.read_csv(f"data/raw/wearable/{participant}_{condition}_right_foot_data.csv", header=0)
    df_mocap = pd.read_csv(f"data/raw/mocap/{participant}_{condition}_mocap.csv",              header=[2, 5, 6])
    df_force = pd.read_csv(f"data/raw/forces/{participant}_{condition}_force.csv",
                           header=10, encoding='shift_jis')
    return dict(left=df_left, right=df_right, mocap=df_mocap, force=df_force,
                mass=mass, participant=participant, condition=condition)
