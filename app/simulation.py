import numpy as np
from libs.vectors import sqrMag, getUnitVectorFast, getUnitVectorFast2, fastMagVec3, fastDistVec3
import libs.physics as phy
import threading
import time
import json
import copy
import pandas as pd
from datetime import datetime
from math import sqrt, floor, pi
import cProfile, pstats, io

class simulation:
    positions = None
    bodies = []
    state = "default"
    minDist = 0.0
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

    # Steps the physics simulation forward using the Leap Frog method
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
                if(r < self.minDist):
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
        #print(f"{self.simTime[0]},\t{np.linalg.norm(self.bodies[2].pos - self.bodies[0].pos) / phy.au}")

    # Iteratively solve for the most accurate velocities
    def setupOrbits(self, iterations, Kp, Ki, Kd, lowestErrorDamp, errorDampSharpness):
        n = len(self.bodies)
        centralBodyIndex = [i for i in range(n) if (self.bodies[i].name in self.bodies[0].orbiting)][0]
        allPeris = [b.peri for b in self.bodies]
        del allPeris[centralBodyIndex]
        allPeris.sort()
        lowestPeri = allPeris[0]
        lastErrors = [0] * n
        integrals = [0] * n
        gracePeriods = [0] * n
        for j in range(iterations):
            for i, body in enumerate(self.bodies):
                if i == centralBodyIndex: continue
                if gracePeriods[i] == 0:
                    a = (body.apo + body.peri) / 2
                    orbitingIndices = [i for i in range(n) if (self.bodies[i].name in body.orbiting)]
                    orbitingBodyMasses = [self.bodies[i].mass for i in orbitingIndices]
                    gravParam = phy.G_const * (sum(orbitingBodyMasses) + body.mass)
                    T = 2 * pi * sqrt(a * a * a / gravParam) # Ideal period in seconds
                    gracePeriods[i] = T / (2.5 * phy.secondsInYear) # Grace period in years
                setupTime = [0,0]
                timestep = (min(body.peri, lowestPeri * 10.0) * phy.secondsInADay) / (phy.au * 8) # Make this smarter
                apoapsis = 0
                periapsis = phy.ly # Just a large distance
                startPos = body.pos - self.bodies[centralBodyIndex].pos
                maxDistance = body.peri / 5.0 # Use eccentricity here somehow
                # If you've travaled too far from your periapsis or apoapsis
                bodiesCopy = copy.deepcopy(self.bodies) # Copy bodies for simulation so that the original are unchanged.
                while fastDistVec3(bodiesCopy[i].pos - bodiesCopy[centralBodyIndex].pos, startPos) > maxDistance or self.floatFromTime(setupTime) < gracePeriods[i]:
                    self.setupStep(bodiesCopy, timestep) # Step forward
                    # Time
                    setupTime[0] = setupTime[0] + floor((setupTime[1] + timestep) / phy.secondsInYear) # Years
                    setupTime[1] = (setupTime[1] + timestep) % phy.secondsInYear # Seconds
                    # Distance
                    r = fastDistVec3(bodiesCopy[i].pos, bodiesCopy[centralBodyIndex].pos)
                    # If the distance is a new periapsis, save it
                    if r < periapsis:
                        periapsis = r
                    if r > apoapsis:
                        apoapsis = r
                # PID controller
                apoError = body.apo - apoapsis
                periError = (body.peri - periapsis) * (body.apo / body.peri) # Peri error more important, so should be weighted
                diff = (abs(apoError) + abs(periError)) / phy.au
                a_ = max(min(1-lowestErrorDamp, 1), 0)
                b_ = max(errorDampSharpness, 0)
                diffError = 1.0 - a_ / (b_ * diff * diff + 1.0)
                error = (apoError + periError) * diffError
                integrals[i] += error * timestep
                deriv = (error - lastErrors[i]) / timestep
                lastErrors[i] = error
                amount = error * Kp + integrals[i] * Ki + deriv * Kd
                # Apply control
                r = fastMagVec3(startPos)
                speed = 500 / r
                unitVecVel = body.vel / fastMagVec3(body.vel)
                body.vel = body.vel + unitVecVel * speed * amount
                print(f"{j}. Body: {self.bodies[i].name}, apoError: {apoError / phy.au}, \tperiError: {periError / phy.au}\tdiffError: {diffError}")

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
            if self.state != "default":
                sendThread.join()
                mainQueue.put('collided')
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
            if self.state != "default":
                sendThread.join()
                mainQueue.put('collided')
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
    orbiting = [""]
    apo = 0.0
    peri = 0.0
    pos = np.array([0.0, 0.0, 0.0])
    vel = np.array([0.0, 0.0, 0.0])
    mass = 1.0 # in solar masses

    def __init__(self, mass, pos, vel):
        self.mass = mass
        self.pos = pos
        self.vel = vel

    # Get the graviational acceleration on this body generated by another body
    def gravAcc(self, other):
        selfToOther = other.pos - self.pos
        sqrR = sqrMag(selfToOther)
        dir = getUnitVectorFast(selfToOther, sqrR)
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
            b.orbiting = data['orbiting'].split(',')
            b.apo = float(data['apo'])
            b.peri = float(data['peri'])
            return b
        except Exception:
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