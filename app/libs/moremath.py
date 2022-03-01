import numpy as np
from libs.vectors import mag2d, mag3d, unitVec3d
from math import sqrt, floor, cos, acos
from scipy.optimize import fsolve
from scipy.linalg import expm, norm

# Ellipse integral with resolution n
def ellipseIntegral(k, n):
    value = 0
    i = 0
    halfpi = np.pi * 0.5
    d = halfpi / n
    while i < halfpi:
        trig = np.sin(i)
        value += np.sqrt(1-k*trig*trig) * d
        i += d
    return value

# arc = a * E(amp|k). k = m. second kind
def ellipseAmplitude(arc, semimajoraxis, k, n):
    value = 0
    amp = 0
    d = 2 * np.pi / n
    while value < arc / semimajoraxis:
        trig = np.sin(amp)
        value += np.sqrt(1-k*trig*trig) * d
        amp += d
    return amp

# Result is given in radians
def angleBetween3d(u, v):
    return acos(np.dot(u,v) / (mag3d(u)*mag3d(v)))

# theta must be in radians
def rotationMatrixAroundAxis3d(axis, theta):
        return expm(np.cross(np.eye(3), axis/norm(axis)*theta))

# Cross product of P and R is used to calculate the normal
def rotate3PointsToPlane(P, Q, R):
    perp = np.cross(P, R)
    perp = unitVec3d(perp)
    up = np.array([0,0,1])
    rotationAxis = np.cross(perp, up)
    angle = angleBetween3d(perp, up) # radians
    M0 = rotationMatrixAroundAxis3d(rotationAxis, angle)
    PlanePQR = [np.dot(M0, p) for p in (P, Q, R)]
    return PlanePQR

# ELLIPSE METHOD
#|P-F| - |Q-F| + |P| - |Q| = 0
def ellipseEquation1(x, args):
    Px, Py, Qx, Qy, Rx, Ry = args
    return (sqrt((Px-x[0])**2 + (Py-x[1])**2)   #   |P - F|
        -sqrt((Qx-x[0])**2 + (Qy-x[1])**2)      # - |Q - F|
        +sqrt(Px**2 + Py**2)                    # + |P|
        -sqrt(Qx**2 + Qy**2))                   # - |Q|

#|P-F| - |R-F| + |P| - |R| = 0
def ellipseEquation2(x, args):
    Px, Py, Qx, Qy, Rx, Ry = args
    return (sqrt((Px-x[0])**2 + (Py-x[1])**2)   #   |P - F|
        -sqrt((Rx-x[0])**2 + (Ry-x[1])**2)      # - |R - F|
        +sqrt(Px**2 + Py**2)                    # + |P|
        -sqrt(Rx**2 + Ry**2))                   # - |R|

# System of non-linear equations ellipseEquation1 and ellipseEquation2
def ellipseEquationSystem(x, args):
    return [ellipseEquation1(x, args), ellipseEquation2(x, args)]

# HYPERBOLA METHOD
#||P-F| - |P|| - ||Q-F| - |Q|| = 0
def hyperbolaEquation1(x, args):
    Px, Py, Qx, Qy, Rx, Ry = args
    return abs(
            mag2d((Px-x[0], Py-x[1]))           #   |P-F|
            -mag2d((Px, Py))                    # - |P|
        ) - abs(
            mag2d((Qx-x[0], Qy-x[1]))           #   |Q-F|
            -mag2d((Qx, Qy))                    # - |Q|
        )

#||P-F| - |P|| - ||R-F| - |R|| = 0
def hyperbolaEquation2(x, args):
    Px, Py, Qx, Qy, Rx, Ry = args
    return abs(
            mag2d((Px-x[0], Py-x[1]))           #   |P-F|
            -mag2d((Px, Py))                    # - |P|
        ) - abs(
            mag2d((Rx-x[0], Ry-x[1]))           #   |R-F|
            -mag2d((Rx, Ry))                    # - |R|
        )

# System of non-linear equations hyperbolaEquation1 and hyperbolaEquation2
def hyperbolaEquationSystem(x, args):
    return [hyperbolaEquation1(x, args), hyperbolaEquation2(x, args)]

# Find second focus based on three points if a known focus lies in (0,0)
def findFocus3PointsEllipse(Px, Py, Qx, Qy, Rx, Ry, searchGridSideRes, gridStepSize, allInfo=False): # If focus is really close to the origin, perhaps reconsider
    maxAttempts = searchGridSideRes * searchGridSideRes
    midpointX = (Px+Qx+Rx)/3
    midpointY = (Py+Qy+Ry)/3
    attempts = 0
    while(attempts < maxAttempts):
        x = midpointX + (attempts % searchGridSideRes - searchGridSideRes * 0.5) * gridStepSize
        y = midpointY + (floor(attempts / searchGridSideRes) - searchGridSideRes) * gridStepSize
        output = fsolve(ellipseEquationSystem, args=((Px, Py, Qx, Qy, Rx, Ry),), x0=[x,y], full_output = True) # Finds the roots of ellipseEquationSystem
        attempts += 1
        if(output[2] == 1): break
    if(attempts < maxAttempts):
        if allInfo:
            return output[0], attempts
        else:
            return output[0]
    raise ModelFitError("3 Points 1 Focus Ellipse", f"Couldn't find F. Attempted to find a suitible ellipse estimate {maxAttempts} times to no avail.")

def findFocus3PointsHyperbola(Px, Py, Qx, Qy, Rx, Ry, searchGridSideRes, stepSize, allInfo=False):
    maxAttempts = searchGridSideRes * searchGridSideRes
    midpointX = (Px+Qx+Rx)/3
    midpointY = (Py+Qy+Ry)/3
    attempts = 0
    while(attempts < maxAttempts):
        x = midpointX + (attempts % searchGridSideRes - searchGridSideRes * 0.5) * stepSize
        y = midpointY + (floor(attempts / searchGridSideRes) - searchGridSideRes) * stepSize
        output = fsolve(hyperbolaEquationSystem, args=((Px, Py, Qx, Qy, Rx, Ry),), x0=[x,y], full_output = True) # Finds the roots of hyperbolaEquationSystem
        attempts += 1
        if(output[2] == 1): break
    if(attempts < maxAttempts):
        if allInfo:
            return output[0], attempts
        else:
            return output[0]
    raise ModelFitError("3 Points 1 Focus Hyperbola", f"Couldn't find F. Attempted to find a suitible hyperbola estimate {maxAttempts} times to no avail.")

# Eccentricity (e) and semi latus rectum (p) of conic section. The search grid is a square with a side of searchGridSide.
def findParams3PointsConic(Px, Py, Qx, Qy, Rx, Ry, searchGridSideRes, stepSize):
    Fx, Fy = 0, 0
    try:
        Fx, Fy = findFocus3PointsEllipse(Px, Py, Qx, Qy, Rx, Ry, searchGridSideRes, stepSize)
    except ModelFitError:
        try:
            Fx, Fy = findFocus3PointsHyperbola(Px, Py, Qx, Qy, Rx, Ry, searchGridSideRes, stepSize)
        except ModelFitError:
            raise ModelFitError("3 Points 1 Focus Conic Section", \
                f"Couldn't fit specified points to neither an ellipse nor a hyperbola. Tried {searchGridSideRes * searchGridSideRes} x 2 times.")
        else:
            c = mag2d((Fx, Fy)) * 0.5
            a = abs(mag2d((Px-Fx, Py-Fy)) - mag2d((Px, Py))) * 0.5
            b = sqrt(c*c - a*a)
    else:
        c = mag2d((Fx, Fy)) * 0.5
        a = (mag2d((Px-Fx, Py-Fy)) + mag2d((Px, Py))) * 0.5
        b = sqrt(a*a - c*c)
    e = c / a
    p = b * b / a
    return e, p

class ModelFitError(Exception):
    def __init__(self, modelType, message="Couldn't fit model of type '{modelType}' to given data."):
        self.modelType = modelType
        self.message = message.format(modelType)
        super().__init__(self.message)

"""
Useful links:
https://math.stackexchange.com/questions/547045/ellipses-given-focus-and-two-points
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html
https://stackoverflow.com/questions/13057022/python-scipy-optimize-using-fsolve-with-multiple-first-guesses
https://stackoverflow.com/questions/40783190/solve-a-system-of-non-linear-equations-in-python-scipy-optimize-fsolve
https://stackoverflow.com/questions/19542801/solving-non-linear-equations-in-python
https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
"""