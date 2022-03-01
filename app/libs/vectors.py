import numpy as np
from math import sqrt

# ALL VECTORS MUST BE NUMPY ARRAYS
def unitVec3d(vec):
    mag = mag3d(vec)
    if mag < 0.001 and mag > -0.001:
        return np.zeros(3)
    return vec / mag

def unitVecFast3d(vec, mag):
    if mag < 0.001 and mag > -0.001:
        return np.zeros(3)
    return vec / mag

def unitVecFastSqr3d(vec, sqrMag):
    mag = sqrt(sqrMag)
    if mag < 0.001 and mag > -0.001:
        return np.zeros(3)
    return vec / mag

# Gets the magnitude of a vector
def mag(vec):
    return np.linalg.norm(vec)

# Magnitude of 2D vector
def mag2d(vec):
    return sqrt(vec[0] * vec[0] + vec[1] * vec[1])

# Squared magnitude of 2D vector
def sqrMag2d(vec):
    return vec[0] * vec[0] + vec[1] * vec[1]

# Distance between two 2D points
def distance2d(origin, target):
    distVec = target - origin
    return sqrt(distVec[0] * distVec[0] + distVec[1] * distVec[1])

# Magnitude of 3D vector
def mag3d(vec):
    return sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])

# Squared magnitude of 3D vector
def sqrMag3d(vec):
    return vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]

# Distance between two 3D points
def distance3d(origin, target):
    distVec = target - origin
    return sqrt(distVec[0] * distVec[0] + distVec[1] * distVec[1] + distVec[2] * distVec[2])
