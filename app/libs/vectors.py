import numpy as np
from math import sqrt

# gets a unit vector
def getUnitVector(vector):
    magnitude = mag(vector)
    if magnitude < 0.001 and magnitude > -0.001:
        return np.array([0,0])
    return np.array([vector[0] / magnitude, vector[1] / magnitude, vector[2] / magnitude])

def getUnitVectorFast(vector, sqrR):
    magnitude = np.sqrt(sqrR)
    if magnitude < 0.001 and magnitude > -0.001:
        return np.array([0,0,0])
    return np.array([vector[0], vector[1], vector[2]] / magnitude)

def getUnitVectorFast2(vector, magnitude):
    if magnitude < 0.001 and magnitude > -0.001:
        return np.zeros(3)
    return np.array(vector) / magnitude

# Gets the magnitude of a vector
def mag(vector):
    return np.linalg.norm(vector)

def fastMagVec3(vector):
    return sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])

# Gets the squared magnitude of a vector
def sqrMag(vector):
    return vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]

def fastDistVec3(origin, target):
    vector = target - origin
    return sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])

# Magnitude of 2D vector
def mag2d(vec2d):
    return sqrt(vec2d[0] * vec2d[0] + vec2d[1] * vec2d[1])
