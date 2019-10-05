import numpy as np
import utils
import constants as c
from kinematics import Kinematics, getInitialPlatformHeight

trans = [0, 0, 0]
rot = [-5, 5, 0]
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
# +5 deg about x - SOUTH
# [16.28929038, 17.3132488, 1.57780453, -1.42523919, -16.94926673, -16.62537064]
# -5 deg about x - NORTH
# [-16.33266239, -17.09930114, -1.54961138, 1.47727662, 17.42187392, 16.48079246]
# +5 deg about y - EAST
# [-10.95487981, -8.22894901, 20.40388639, 18.90780741, -8.19305569, -11.24629653]
# -5 deg about y - WEST
# [11.29125504, 7.9723512, -20.17743784, -18.74528518, 7.99369953, 11.34146758]
# +5 deg about x-y-q1 - SOUTH-EAST
# [4.63656683, 9.32163705, 22.11165996, 17.38644301, -26.17547899, -28.40410827]
# -5 deg about x-y-q1 - NORTH-WEST
# [-5.31094234, -8.30387182, -21.81135837, -17.19976703, 25.61911003, 28.82278151]
# +5 deg about x-y-q2 - SOUTH-WEST
# [29.00981416, 25.38119401, -18.52966373, -20.24932271, -8.26197035, -5.21160833]
# -5 deg about x-y-q2 - NORTH-EAST
# [-27.53319234, -26.51660974, 18.74628217, 20.49904252, 9.35061747, 4.89456401]





