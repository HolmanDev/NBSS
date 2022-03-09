import os
import numpy as np
from libs.vectors import mag3d, sqrMag3d, unitVec3d, unitVecFastSqr3d, distance3d
import libs.physics as phy
import threading
import time
import json
import copy
import pandas as pd
from datetime import datetime
import math
from math import sqrt, floor
import cProfile, pstats, io

class simulation:
    positions = None
    bodies = []
    state = "default"
    collisionDistFactor = 1.5
    queuedPacket = None
    collisionLogPath = ""
    startTime = pd.to_datetime('2000-01-01')
    simTime = [2000,0] # year, fraction
    lastRebaseYear = 2000
    positionRebaseInterval = 10000 # in years
    pathLength = 10000

    def floatFromTime(self, time):
        return time[0] + time[1] / phy.secondsInYear

    # Moves anchorBody to (0,0,0) and moves every body according to their relative positions to anchorBody.
    def rebasePosition(self, anchorBody):
        for body in self.bodies:
            if body is anchorBody: continue
            body.pos = body.pos - anchorBody.pos
            body.vel = body.vel - anchorBody.vel
        anchorBody.pos = np.zeros(3)
        anchorBody.vel = np.zeros(3)

    def collisionHandler(self, body1, body2):
        os.makedirs(self.collisionLogPath, exist_ok=True)
        now = datetime.now()
        filename = f"{self.collisionLogPath}collisions_{now.year}-{now.month}-{now.day}T{now.hour}-{now.minute}-{now.second}.txt"
        with open(filename, "w+") as f:
            timeValue = self.simTime[0] + self.simTime[1] / 31557600.0
            output = f"{body1.name} and {body2.name} collided!\n{timeValue}\n"
            for body in self.bodies:
                output = output + f"\"{body.name}\","
            output = output[:-1] + "\n"
            for n in range(len(self.bodies)):
                output = output + f"BODY-{n}{{{self.bodies[n].pos},{self.bodies[n].vel}}}\n" 
            f.write(output)

    # Steps the physics simulation forward using the leapfrog method
    def step(self, timestep):
        # Position at t+0.5
        for body in self.bodies:
            body.pos = body.pos + 0.5 * timestep * body.vel # x_n+0.5
        # Only calculate distances and directions once per pair of bodies
        for i, body1 in enumerate(self.bodies):
            for body2 in self.bodies[i + 1:]:
                selfToOther = body2.pos - body1.pos
                r = sqrt(selfToOther[0] * selfToOther[0] + selfToOther[1] * selfToOther[1] + selfToOther[2] * selfToOther[2])
                accNoMass = (phy.G_const / (r * r * r)) * selfToOther
                # Velocity at t+1
                body1.vel = body1.vel + timestep * body2.mass * accNoMass
                body2.vel = body2.vel - timestep * body1.mass * accNoMass
                # Handle collision
                collisionDist = body1.radius + body2.radius
                if(r < collisionDist * self.collisionDistFactor):
                    self.state = "collided"
                    self.collisionHandler(body1, body2)
                    return
        # Position at t+1
        for body in self.bodies:
            body.pos = body.pos + 0.5 * timestep * body.vel #x_n+1

    # Steps the physics simulation forward using the Leap Frog method
    def setupStep(self, bodies, timestep):
        # Position at t+0.5
        for body in bodies:
            body.pos = body.pos + 0.5 * timestep * body.vel # x_n+0.5
        # Only calculate distances and directions once per pair of bodies
        for i, body1 in enumerate(bodies):
            for body2 in bodies[i + 1:]:
                selfToOther = body2.pos - body1.pos
                r = sqrt(selfToOther[0] * selfToOther[0] + selfToOther[1] * selfToOther[1] + selfToOther[2] * selfToOther[2])
                accNoMass = (phy.G_const / (r * r * r)) * selfToOther
                # Velocity at t+1
                body1.vel = body1.vel + timestep * body2.mass * accNoMass
                body2.vel = body2.vel - timestep * body1.mass * accNoMass
        # Position at t+1
        for body in bodies:
            body.pos = body.pos + 0.5 * timestep * body.vel #x_n+1

    def leap(self, steps, timestep):
        for _ in range(steps):
            self.step(timestep)
            if (self.state == "collided"): break
        secondsDelta = timestep * steps
        yearsDelta = floor((self.simTime[1] + secondsDelta) / phy.secondsInYear)
        self.simTime[0] = self.simTime[0] + yearsDelta
        self.simTime[1] = (self.simTime[1] + secondsDelta) % phy.secondsInYear

    # Brings the errors to rest in a valley
    def valleyErrorCalc(self, error1, error2, lowestErrorDamp, errorDampSharpness):
        diff = (abs(error1) + abs(error2)) / phy.au
        a = max(min(1-lowestErrorDamp, 1), 0)
        b = max(errorDampSharpness, 0)
        diffError = 1.0 - a / (b * diff * diff + 1.0)
        error = (error1 + error2) * diffError
        return error

    def findPQR(self, bodies, bodyIndex, centralBodyIndex, timestep, approxTime):
        P, Q, R = np.empty(3)
        setupTime = [0,0]
        searchingForQ = True
        P = bodies[bodyIndex].pos - bodies[centralBodyIndex].pos # First point found
        while self.floatFromTime(setupTime) < approxTime:
            # Search for second point
            if(searchingForQ and self.floatFromTime(setupTime) > approxTime * 0.5):
                Q = bodies[bodyIndex].pos - bodies[centralBodyIndex].pos # Second point found
                searchingForQ = False
            # Simulate
            self.setupStep(bodies, timestep)
            # Time
            setupTime[0] = setupTime[0] + floor((setupTime[1] + timestep) / phy.secondsInYear) # Years
            setupTime[1] = (setupTime[1] + timestep) % phy.secondsInYear # Seconds
        R = bodies[bodyIndex].pos - bodies[centralBodyIndex].pos # Third point found
        return P, Q, R

    def findCOM(self, bodies):
        totalMass = 0
        com = np.zeros(3)
        for b in bodies:
            totalMass += b.mass
            com = com + b.pos * b.mass
        com = com / totalMass
        return com
    
    def findCOMVelocity(self, bodies):
        totalMass = 0
        comVel = np.zeros(3)
        for b in bodies:
            totalMass += b.mass
            comVel = comVel + b.vel * b.mass
        comVel = comVel / totalMass
        return comVel

    # Iteratively solve for the most accurate initial velocities
    def setupOrbits(self, iterations, Kp, Ki, Kd, lowestErrorDamp, errorDampSharpness, doPID=True):
        from libs.moremath import rotate3PointsToPlane, findParams3PointsEllipse, ModelFitError
        n = len(self.bodies)
        centralBodyIndex = [i for i in range(n) if (self.bodies[i].name in self.bodies[i].refBodies)][0]
        allPeris = [b.idealPeri for i,b in enumerate(self.bodies) if (i != centralBodyIndex and b.idealPeri > 0)]
        allPeris.sort()
        lowestPeri = allPeris[0]
        for body in self.bodies:
            refBodies = [b for b in self.bodies if (b.name in body.refBodies)]
            comVelocity = self.findCOMVelocity(refBodies)
            body.vel = body.vel + comVelocity
            print(f"bodyVelocity: {mag3d(body.vel)}")
        # Use PID controller to arrive at optimal initial velocities
        if(not doPID): return
        lastErrors = [0] * n
        integrals = [0] * n
        for j in range(iterations):
            for i, body in enumerate(self.bodies):
                if i == centralBodyIndex: continue
                # SEARCH PREREQUISITES
                T = body.orbitTime(self.bodies)
                approxTime = T / (10.0 * phy.secondsInYear) # In years
                timestep = (min(body.idealPeri, lowestPeri * 100.0) * phy.secondsInADay) / (phy.au * 64) #! Make this smarter
                bodiesCopy = copy.deepcopy(self.bodies) # Copy bodies for simulation so that the original are unchanged.
                # SAMPLE POINTS
                P, Q, R = self.findPQR(bodiesCopy, i, centralBodyIndex, timestep, approxTime) # Find sample points P, Q and R
                (Px, Py, _), (Qx, Qy, _), (Rx, Ry, _) = rotate3PointsToPlane(P, Q, R) # Rotate the points around the origin onto a plane
                # ESTIMATE ECCENTRICITY AND SEMI-LATUS RECTUM
                # Try to find e and p. If ellipse fitting fails, set velocity to escape velocity and go down until it the ellipse fitting succeeds.
                try:
                    e, p = findParams3PointsEllipse(Px, Py, Qx, Qy, Rx, Ry, 30, body.idealPeri / (20 * phy.au))
                except ModelFitError:
                    bodiesCopy = copy.deepcopy(self.bodies)
                    refBodies = [b for b in self.bodies if (b.name in body.refBodies)]
                    refBodiesMass = sum([b.mass for b in refBodies])
                    com = self.findCOM(refBodies) # center of mass
                    d = mag3d(body.pos - com)
                    escapeSpeed = sqrt(2 * phy.G_const * refBodiesMass / d)
                    velMag = mag3d(body.vel)
                    comVelocity = self.findCOMVelocity(refBodies)
                    bodiesCopy[i].vel = (0.75 * escapeSpeed / velMag) * body.vel + comVelocity
                    e, p = catchOrbit(0.85, bodiesCopy, i, centralBodyIndex, timestep, approxTime) #! Make factor variable in the UI
                # Catch the orbit by slowly lowering the speed by a factor
                def catchOrbit(factor, bodies, i, centralBodyIndex, timestep, approxTime): #! Should regulate error relative to moving COM!!!!!
                    print("Trying to catch")
                    velocity = bodies[i].vel
                    P, Q, R = self.findPQR(bodies, i, centralBodyIndex, timestep, approxTime)
                    (Px, Py, _), (Qx, Qy, _), (Rx, Ry, _) = rotate3PointsToPlane(P, Q, R)
                    try:
                        e, p = findParams3PointsEllipse(Px, Py, Qx, Qy, Rx, Ry, 30, body.idealPeri / (20 * phy.au))
                    except ModelFitError:
                        bodies = copy.deepcopy(self.bodies)
                        bodies[i].vel = factor * velocity # Lower velocity
                        print(f"velocity: {mag3d(bodies[i].vel)}")
                        catchOrbit(factor, bodies, i, centralBodyIndex, timestep, approxTime)
                    else:
                        return e, p
                #e, p = findParams3PointsConic(Px, Py, Qx, Qy, Rx, Ry, 30, body.idealPeri / (20 * phy.au))  #!Hyperbolic fitting bad. Try to get eccentricity to below 1.
                #! ^ Yeah, switch this to just try to find the ellipse. If it doesn't, set the speed to the calculated escape velocity and go down 5% at a time.
                #! The escape velocity can be calculated using the formula sqrt(2 * G * M / d) where M is the mass of the primary body and d is the distance
                print(f"Eccentricity: {e}, semi-latus rectum: {p / phy.au}")
                # PID ERROR HANDLING
                eError = (body.idealEcc - e) * (body.idealSemiLatRect / (body.idealEcc * 1000000.0)) #! Eccentricty error should be weighted based on orbit size
                pError = body.idealSemiLatRect - p
                error = self.valleyErrorCalc(eError, pError, lowestErrorDamp, errorDampSharpness)
                integrals[i] += error * timestep
                deriv = (error - lastErrors[i]) / timestep
                lastErrors[i] = error
                amount = error * Kp + integrals[i] * Ki + deriv * Kd
                # APPLY CONTROL
                r = distance3d(body.pos, self.bodies[centralBodyIndex].pos)
                speed = 5000 / r
                unitVecVel = unitVec3d(body.vel)
                body.vel = body.vel + unitVecVel * speed * amount
                print(f"{j}. Body: {self.bodies[i].name}, eError: {body.idealEcc-e}, \tpError: {pError/phy.au}\tError: {error/phy.au}, \tspeed: {speed}")

    # Returns an array of the history of positions plus the new ones from the generated data, 
    # containing [snaps] snaps each containing data from [stepsPerSnap] steps taken with a timestep of [timestep].
    def genPositions(self, snaps, stepsPerSnap, timestep, oldPositions = None):
        positions = [np.empty((3, snaps)) for _ in range(len(self.bodies))]
        for i in range(snaps):
            self.leap(stepsPerSnap, timestep)
            if self.state == "collided": return
            for n, body in enumerate(self.bodies):
                positions[n][:, i] = np.copy(body.pos)
        if oldPositions is not None:
            positions = [np.append(oldPositions[n], positions[n], 1) for n in range(len(self.bodies))]
        return positions

    def posGenLoop(self, snaps, stepsPerSnap, timestep, delay, startTime, mainQueue, pipeConnection):
        self.startTime = startTime
        self.simTime = [startTime.year, (startTime.month * 30.5 + startTime.day) / 365.25]
        self.lastRebaseYear = startTime.year
        sendThread = threading.Thread(target=self.sendPosition, name='sender', args=(pipeConnection, delay)) # The 2 should be replaced with something
        sendThread.start()
        while(self.state == "default"):
            positions = self.genPositions(snaps, stepsPerSnap, timestep, self.positions)
            mainQueue.put(('vis', 'bodies', self.bodies))
            mainQueue.put(('main', 'bodies', self.bodies))
            if self.state != "default": #! Double negative. Fix.
                sendThread.join()
                mainQueue.put(('main', 'collided', ''))
                break
            else:
                # Cut tail
                if(len(positions[0][0,:]) > self.pathLength):
                    positions = [bodyPositions[:, -self.pathLength:] for bodyPositions in positions]
                #Store new positions
                self.positions = positions
                # Rebase
                if(self.simTime[0] - self.lastRebaseYear > self.positionRebaseInterval):
                    self.lastRebaseYear = self.simTime[0]
                    self.rebasePosition(self.bodies[0]) # Is this always right?
        pipeConnection.close()

    def sendPosition(self, pipeConnection, delay):
        while(True):
            try:
                pipeConnection.send([copy.deepcopy(self.positions), len(self.bodies), [b.name for b in self.bodies], self.state])
            finally:        
                if self.state != "default":
                    break
            time.sleep(delay)

    # Returns a packet of generated data, containing [snapsPerPacket] snaps each containing data 
    # from [stepsPerSnap] steps taken with a timestep of [timestep].
    def genPacket(self, snapsPerPacket, stepsPerSnap, timestep):
        newPacket = packet(
            [b.name for b in self.bodies],
            [[np.array([0,0,0]) for i in range(snapsPerPacket)] for n in range(len(self.bodies))],  # positions
            [[np.array([0,0,0]) for i in range(snapsPerPacket)] for n in range(len(self.bodies))]   # velocities
        ) # Copy already existing packet template instead of performance gain?
        for i in range(snapsPerPacket):
            self.leap(stepsPerSnap, timestep) # Leaps and thereby executes all the steps in a snap
            if self.state == "collided": return
            for n, body in enumerate(self.bodies):
                newPacket.positions[n][i] = np.copy(body.pos)
                newPacket.velocities[n][i] = np.copy(body.vel)
        return newPacket

    def packetGenLoop(self, snapsPerPacket, stepsPerSnap, timestep, delay, startTime, mainQueue, pipeConnection):
        self.startTime = startTime
        self.simTime = [startTime.year, (startTime.month * 30.5 + startTime.day) / 365.25]
        self.lastRebaseYear = startTime.year
        sendThread = threading.Thread(target=self.sendPacket, name='sender', args=(pipeConnection, delay))
        sendThread.start()
        while(True):
            packet = self.genPacket(snapsPerPacket, stepsPerSnap, timestep)
            mainQueue.put(('vis', 'bodies', self.bodies))
            mainQueue.put(('main', 'bodies', self.bodies))
            if self.state != "default":
                sendThread.join()
                mainQueue.put(('main', 'collided', ''))
                break
            else:
                # Store new packet
                self.queuedPacket = packet
                # Rebase
                if(self.simTime[0] - self.lastRebaseYear > self.positionRebaseInterval):
                    self.lastRebaseYear = self.simTime[0] # Am i copying a reference or a value here?
                    self.rebasePosition(self.bodies[0]) # Is this always right?
        pipeConnection.close()

    def sendPacket(self, pipeConnection, delay):
        while(True):          
            if(self.queuedPacket is not None): # Is this still needed?
                pipeConnection.send([self.queuedPacket, self.simTime, self.state]) # The packet sent is a reference here!
            if self.state != "default":
                break
            time.sleep(delay)

class body:
    name = ""  
    mass = 1.0 # in solar masses
    radius = 0.0
    refBodies = [""] # Names of bodies that this body orbits around
    idealApo = 0.0
    idealPeri = 0.0
    idealEcc = 0.0
    idealSemiLatRect = 0.0
    pos = np.array([0.0, 0.0, 0.0])
    vel = np.array([0.0, 0.0, 0.0])

    def __init__(self, mass, pos, vel):
        self.mass = mass
        self.pos = pos
        self.vel = vel

    def calcApoAndPeri(self):
        a = self.idealSemiLatRect / (1 - self.idealEcc*self.idealEcc)
        if a < 0: 
            self.idealApo = -1
            self.idealPeri = -1
            return
        c = self.idealEcc * a
        self.idealApo = a + c
        self.idealPeri = a - c

    def orbitTime(self, allBodies):
        a = self.idealSemiLatRect / (1.0 - self.idealEcc*self.idealEcc)
        refBodyIndices = [i for i in range(len(allBodies)) if (allBodies[i].name in self.refBodies)]
        refBodyMasses = [allBodies[i].mass for i in refBodyIndices]
        gravParam = phy.G_const * (sum(refBodyMasses) + self.mass)
        T = 2 * math.pi * sqrt(a * a * a / gravParam) # Ideal period in seconds
        return T

    # Get the graviational acceleration on this body generated by another body.
    def gravAcc(self, other): #! Untested
        selfToOther = other.pos - self.pos
        sqrR = sqrMag3d(selfToOther)
        dir = unitVecFastSqr3d(selfToOther, sqrR)
        return dir * (phy.G_const * other.mass / sqrR)

    # dirBySqr = dir / sqrR
    def gravAccEco(self, other, dirBySqr):
        return dirBySqr * (phy.G_const * other.mass)

    # Get the graviational force acting on the body
    def gravForce(r, m1, m2):
        return phy.G_const * (m1 * m2) / (r * r)

    def parseJSON(text):
        try:
            data = json.loads(text)
            b = body(float(data['mass']),\
                np.array([float(data['x_pos']),float(data['y_pos']),float(data['z_pos'])]),\
                np.array([float(data['x_vel']),float(data['y_vel']),float(data['z_vel'])]))
            b.name = data['name']
            b.radius = float(data['radius'])
            b.refBodies = data['refBodies'].split(',')
            b.idealEcc = float(data['ecc'])
            b.idealSemiLatRect = float(data['semiLatRect'])
            b.calcApoAndPeri()
            return b
        except Exception: #! Should catch a specified type of error
            raise ValueError('Could not interpret file as body data in json format')

class packet():
    def __init__(self, names, positions, velocities):
        self.names = names
        self.positions = positions
        self.velocities = velocities

    def __repr__(self):
        output = ""
        for name in self.names:
            output = output + "\"" + name + "\","
        output = output[:-1] + "\n"
        for n in range(len(self.positions)):
            output = output + "BODY-" + str(n) + "\n{\n"
            for i in range(len(self.positions[0])):
                output = output + str(i) + ":" + str(self.positions[n][i]) + "," + str(self.velocities[n][i]) + "\n"
            output = output + "}\n"
        return output

    def getNumBodies(text):
        n = text.count('{')
        return n

    def getNumSnaps(text):
        a = text.split('{')
        n = a[1].count(':')
        return n

    # From text to packet
    def parse(text, numSnaps, numBodies):
        # Handle names
        k = text.splitlines()[2] # select the third row, containing the names
        names = k.split("\"")[1::2] # slice notation
        # Do data stuff
        a = text.split("{")
        output = [[[] for i in range(numSnaps)] for n in range(numBodies)]
        for n in range(0, numBodies):
            bodyData = a[n + 1].split("}")[0]
            # We have now singled out a body
            p = bodyData.split(":")
            for i in range(0, len(p) - 1):
                data = p[i + 1].rsplit("]", 1)[0]
                data = data.replace("[", "").replace("]", "")
                # Now we have singled out a snap from a body
                data = data.split(",")
                # Handle position
                posStr = data[0].split()
                pos = np.array([float(posStr[0]), float(posStr[1]), float(posStr[2])])
                # Handle velocity
                velStr = data[1].split()
                vel = np.array([float(velStr[0]), float(velStr[1]), float(velStr[2])])
                # Make output
                output[n][i] = [pos, vel, names[n]]
        return output