import numpy as np
import utils
import constants as c
from kinematics import Kinematics, getInitialPlatformHeight

trans = [0, 0, 0]
rot = [-10, 0, 0]
base_anchor = utils.getAllBaseAnchor()
platform_anchor = utils.getAllPlatformAnchor()
beta_k = c.BETA_K

kin = Kinematics()
alpha = []
for i in range(6):
    alpha.append(kin.inverse_kinematics(trans, rot, base_anchor[i], platform_anchor[i], beta_k[i]))

# print(base_anchor)
# print(platform_anchor)
# print(beta_k)
print(np.array(alpha)- np.array(c.REST_ANGLE))
