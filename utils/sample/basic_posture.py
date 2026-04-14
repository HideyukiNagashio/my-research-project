# basic_posture.py

import numpy as np
from gait_feedback.hardware.natnet_wrapper import NatNetWrapper
from gait_feedback.angle import CalculateAngle
from gait_feedback.config import RIGIDBODY_NO, R_TOE_INDEX, L_TOE_INDEX


def measure_basic_posture(wrapper: NatNetWrapper, verbose: bool = True):

    sorted_ids, pos = wrapper.get_latest_positions()
    if pos.size == 0:
        raise RuntimeError("剛体の座標を取得エラー")

    angle = CalculateAngle().angles(pos)[:, None, :]

    r_ht = pos[R_TOE_INDEX -1, 1] * 1000
    l_ht = pos[L_TOE_INDEX - 1, 1] * 1000

    if verbose:
        print("\n--- 取得した剛体 ID & 座標 ---")
        for rid, (x, y, z) in zip(sorted_ids, pos):
            print(f"ID {rid:>2}:  X={x:+.4f}  Y={y:+.4f}  Z={z:+.4f}")

        expected_ids = list(range(1, RIGIDBODY_NO + 1))
        missing_ids = [i for i in expected_ids if i not in sorted_ids]
        extra_ids = [i for i in sorted_ids if i not in expected_ids]

        print(f"読み取れた剛体ID: {sorted(sorted_ids)}")
        if missing_ids:
            print(f"読み取れていない剛体ID: {missing_ids}")
        else:
            print("13剛体を正常に読み取り")
        if extra_ids:
            print(f"エラー: {extra_ids}")

        print("\n----- 基本関節角度 (deg) [xy, yz, zx] -----")
        joint_names = ["Trunk", "R-Hip", "L-Hip", "R-Knee",
                       "L-Knee", "R-Ankle", "L-Ankle"]
        
        for name, ang in zip(joint_names, angle[:, 0, :]):
            xy, yz, zx = ang
            print(f"{name:<8}:  xy={xy:6.2f}  yz={yz:6.2f}  zx={zx:6.2f}")
        print("\nR拇指床間距離 {:.1f} mm".format(r_ht))
        print("L拇指床間距離 {:.1f} mm".format(l_ht))
        print("-" * 40)

    return angle, r_ht, l_ht