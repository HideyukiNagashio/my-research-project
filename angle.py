#angle.py

import numpy as np
from numpy import linalg as LA

from gait_feedback.config import (
    JOINT_NO, HEAD_INDEX, TRUNK_INDEX, SACRAL_INDEX,
    R_ILIUM_INDEX, L_ILIUM_INDEX,
    R_GREATER_TROCHANTER_INDEX, L_GREATER_TROCHANTER_INDEX,
    R_KNEE_INDEX, L_KNEE_INDEX,
    R_MALLEOLUS_INDEX, L_MALLEOLUS_INDEX,
    R_TOE_INDEX, L_TOE_INDEX,
)

class CalculateAngle:

    @staticmethod
    def _vec(A, B, C):
        return np.stack((A - B, C - B), axis=0)

    @staticmethod
    def _angle2d(a, b):
        inner = np.dot(a, b)
        norm  = LA.norm(a) * LA.norm(b)
        return np.degrees(np.arccos(np.clip(inner / norm, -1, 1)))

    def _angle3d(self, v):
        deg   = np.zeros(3)
        pairs = [
            (slice(None, 2),            0),  # xy
            (slice(1, None),            1),  # yz
            ((2, 0),                    2)   # zx
        ]
        
        for i, (sl, _) in enumerate(pairs):
            a = v[0][sl] if isinstance(sl, slice) else v[0][list(sl)]
            b = v[1][sl] if isinstance(sl, slice) else v[1][list(sl)]
            S = 0.5 * (b[0] * a[1] - b[1] * a[0])  
            ang = self._angle2d(a, b)
            deg[i] = 360 - ang if S < 0 else ang
        return deg

    def angles(self, pos):
        pos = pos[:, [0, 2, 1]]   # XYZ → X-Z-Y
        pos[:, 0] *= -1           # X 軸反転
        out = np.zeros((JOINT_NO, 3))
    
        idxs = [
            # 1. Trunk
            (HEAD_INDEX - 1, TRUNK_INDEX - 1, SACRAL_INDEX - 1),
            # 2. R-Hip
            (R_ILIUM_INDEX - 1, R_GREATER_TROCHANTER_INDEX - 1, R_KNEE_INDEX - 1),
            # 3. L-Hip
            (L_ILIUM_INDEX - 1, L_GREATER_TROCHANTER_INDEX - 1, L_KNEE_INDEX - 1),
            # 4. R-Knee
            (R_GREATER_TROCHANTER_INDEX - 1, R_KNEE_INDEX - 1, R_MALLEOLUS_INDEX - 1),
            # 5. L-Knee
            (L_GREATER_TROCHANTER_INDEX - 1, L_KNEE_INDEX - 1, L_MALLEOLUS_INDEX - 1),
            # 6. R-Ankle
            (R_KNEE_INDEX - 1, R_MALLEOLUS_INDEX - 1, R_TOE_INDEX - 1),
            # 7. L-Ankle
            (L_KNEE_INDEX - 1, L_MALLEOLUS_INDEX - 1, L_TOE_INDEX - 1),
        ]

        for i, (a, b, c) in enumerate(idxs):
            out[i] = self._angle3d(self._vec(pos[a], pos[b], pos[c]))
        return out