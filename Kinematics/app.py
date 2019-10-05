from kinematics import Kinematics, getInitialPlatformHeight
import utils

trans = [0, 0, 0]
rot = [10, 0, 0]
base_anchor = [93.07, 53.95, 0]
platform_anchor = [54.26, 78.95, 0]
beta_k = 120.1

kin = Kinematics()

alpha = kin.inverse_kinematics(trans, rot, base_anchor, platform_anchor, beta_k)
print(alpha)