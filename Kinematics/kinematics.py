import math
import copy
import numpy as np
from pyquaternion import Quaternion

#  input: [surge, sway, heave],[roll, pitch, yaw]
#  output: motor angle alpha

# Global var for generic length and position in mm
leg_len = 182
horn_len = 24
initial_base_anchor_pos = np.array([93.07, 53.95, 0])
initial_platform_anchor_pos = np.array([54.26, 78.95, 177.676])

def getInitialPlatformHeight():
    return 177.676
    # return (
    #     math.sqrt(
    #         leg_len**2 + horn_len**2 
    #         - (initial_platform_anchor_pos[0] - initial_base_anchor_pos[0])**2 
    #         - (initial_platform_anchor_pos[1] - initial_base_anchor_pos[1])**2
    #     )
    # )

# Global var for reference frame and length in mm -> need to modify
base_pos = np.array([0, 0, 0])
platform_pos = np.array([0, 0, getInitialPlatformHeight()])

class Kinematics:
    def __init__(self):
        pass

    def getQuaternionParams(self, roll, pitch, yaw):
        sin_roll = math.sin(roll * 0.5)
        sin_pitch = math.sin(pitch * 0.5)
        sin_yaw = math.sin(yaw * 0.5)
        cos_roll = math.cos(roll * 0.5)
        cos_pitch = math.cos(pitch * 0.5)
        cos_yaw = math.cos(yaw * 0.5)

        w = cos_yaw * cos_pitch * cos_roll + sin_yaw * sin_pitch * sin_roll
        x = cos_yaw * cos_pitch * sin_roll - sin_yaw * sin_pitch * cos_roll
        y = sin_yaw * cos_pitch * sin_roll + cos_yaw * sin_pitch * cos_roll
        z = sin_yaw * cos_pitch * cos_roll - cos_yaw * sin_pitch * sin_roll

        return np.array([w, x, y, z])
    
    """
        trans: [x,y,z] relative translational change
        rot: [roll, pitch, yaw] relative rotational change
        base_anchor: [x,y,z] leg anchor on base relative to base_pos
        platform_anchor: [x,y,z] leg anchor on platform relative to platform_pos
    """
    def inverse_kinematics(self, trans, rot, base_anchor, platform_anchor, beta_k):
        local_trans = np.array(copy.copy(trans))
        local_rot = np.radians(np.array(copy.copy(rot)))
        Bk = np.array(base_anchor)
        Pk = np.array(platform_anchor)
        
        # surge, sway, heave = local_trans
        roll, pitch, yaw = local_rot
        qt_params = self.getQuaternionParams(roll, pitch, yaw)
        qt = Quaternion(qt_params).normalised

        # 3x3 np-array
        Lk = (platform_pos + local_trans) + qt.rotate(Pk) - Bk
        Lk_len = np.linalg.norm(Lk)

        ek = 2 * horn_len * Lk[2]
        fk = 2 * horn_len * ((math.cos(beta_k) * Lk[0]) + (math.sin(beta_k) * Lk[1]))
        gk = (Lk_len)**2 - ((leg_len)**2 - (horn_len)**2)

        alpha_k_rad = math.asin(gk/math.sqrt((ek)**2 + (fk)**2)) - math.atan2(fk, ek)
        alpha_k_deg = math.degrees(alpha_k_rad)

        return alpha_k_deg
