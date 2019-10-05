import math
import numpy as np
import constants as c

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return [rho, phi]

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return [x, y]

def cyl2cart_3D(r, phi, h):
    x, y = pol2cart(r, phi)
    return [x, y, h]

def sph2cart_3D(sph_coords):
    r, phi, theta = sph_coords
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return [x, y, z]

def cart2sph_3D(cart_coords):
    x,y,z = cart_coords
    print(x,y,z)
    XsqPlusYsq = x**2 + y**2
    r = math.sqrt(XsqPlusYsq + z**2)
    theta = math.atan2(z,math.sqrt(XsqPlusYsq))
    phi = math.atan2(y,x)
    return [r, phi, theta]

def getAllBaseAnchor():
    angles = np.radians(c.BASE_ANCHOR_ANGLE)
    res = []
    for angle in angles:
        res.append(pol2cart(c.BASE_ANCHOR_RADIUS, angle))
    return np.array(res)

def getAllPlatformAnchor():
    angles = np.radians(c.PLATFORM_ANCHOR_ANGLE)
    res =[]
    for angle in angles:
        res.append(pol2cart(c.PLATFORM_ANCHOR_RADIUS, angle))
    return np.array(res)