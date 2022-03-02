import numpy as np
import libs.vectors as vec
import libs.physics as phy
import simulation as sim
from matplotlib import pyplot as plt
from matplotlib import animation
import os
import threading
import time
import copy

class visualization:
    lockedBody = "" # name of the body in the bodies array. "" = no body
    pipeData = []
    uiBackend = None
    activeBodies = []

    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.set_box_aspect((1,1,1))

    def awaitMessage(self, queue, delay):
        while(True):
            message = queue.get() # Create standard format for message
            receiver = message[0]
            if(receiver == "vis"):
                title = message[1]
                content = message[2]
                if(title == "bodies"):
                    self.activeBodies = content
            else:
                queue.put(message)
            time.sleep(delay)

    def startAwaitMessageThread(self, queue, delay):
        awaitMessageThread = threading.Thread(target=self.awaitMessage, name='awaiter', args=(queue, delay))
        awaitMessageThread.start()

    # Create orbit paths
    def createOrbitPaths(self, bodies):
        self.orbitPaths = [0] * len(bodies)
        for n in range(0, len(bodies)):
            try:
                self.orbitPaths[n] = self.ax.plot(bodies[n].pos[0], bodies[n].pos[1], bodies[n].pos[2])[0]
            except: #! Should catch a specified type of error
                self.orbitPaths[n] = self.ax.plot(0,0,0)[0]

    # Sets the viewing boundries of the 3D plot
    def setBoundries(self, center, zoom):
        self.ax.set_xlim3d([-zoom * phy.au + center[0], zoom * phy.au + center[0]])
        self.ax.set_xlabel('X')
        self.ax.set_ylim3d([-zoom * phy.au + center[1], zoom * phy.au + center[1]])
        self.ax.set_ylabel('Y')
        self.ax.set_zlim3d([-zoom * phy.au + center[2], zoom * phy.au + center[2]])
        self.ax.set_zlabel('Z')

    def markOrigin(self, mark):
        plt.plot(0,0,0, marker=mark)

    def getPipeData(self, pipeConnection, delay):
        while True:
            try:
                while(pipeConnection.poll()): #! Might be at risk of overload
                    self.pipeData = pipeConnection.recv()
            finally:
                time.sleep(delay)

    # Update the plot animation. num is the frame
    def animUpdate(self, num):
        if(self.pipeData == []):
            return

        state = self.pipeData[3]
        if(state == "collided"):
            return

        cachedPositions = self.pipeData[0]
        names = self.pipeData[2]
        for n in range(self.pipeData[1]):
            offsetPositions = cachedPositions[n]
            numBodies = len(cachedPositions)
            numSnaps = len(offsetPositions[0, :])
            origin = np.zeros([3, numSnaps])
            if(self.lockedBody == "" or self.lockedBody not in names): # Empty locked body => com
                com = np.zeros([3, numSnaps])
                masses = []
                for i in range(numBodies):
                    masses.append(next(b.mass for b in self.activeBodies if b.name == names[i]))
                    com = com + cachedPositions[i] * masses[i]
                com = com / sum(masses)
                origin = com
            else:
                origin = cachedPositions[names.index(self.lockedBody)]
            offsetPositions = offsetPositions - origin
            XnY = offsetPositions[0:2, :]
            Z = offsetPositions[2, :]
            self.orbitPaths[n].set_data(XnY)
            self.orbitPaths[n].set_3d_properties(Z)

    # Visualize the log. num is the frame
    def visualizeLog(self, path, showNth):
        fileNames = [x for x in os.listdir(path) if x.startswith("packet_")]
        if(fileNames == []): return

        # Get information about the system from the first packet
        with open(path + fileNames[0], "r") as f:
            firstPacket = f.read()
            numSnaps = sim.packet.getNumSnaps(firstPacket)
            numBodies = sim.packet.getNumBodies(firstPacket)
            firstData = sim.packet.parse(firstPacket, numSnaps, numBodies)
            
        names = [b[0][2] for b in firstData]
        # Extract and store positions
        positions = [np.empty([3, 0]) for _ in range(numBodies)]
        fileNames.sort(key = lambda x: int(x.split("_")[1]))    
        #fileNames = fileNames[:1000] # Show only the first 1000 logs. #! Make variable
        fileNames = fileNames[0::showNth] # Only show every nth packet
        for fileName in fileNames:
            with open(path + fileName, "r") as f:
                data = sim.packet.parse(f.read(), numSnaps, numBodies)
                for n in range(numBodies):
                    posData = [np.array(data[n][i][0]) for i in range(len(data[n]))] #body, snap, pos or vel
                    newPositions = np.empty([3, numSnaps])
                    for i in range(numSnaps):
                       newPositions[:, i] = posData[i]
                    positions[n] = np.append(positions[n], newPositions, axis=1)

        # Create temporary bodies and make their orbit paths
        bodies = [sim.body(1, np.array(positions[n][:, 0]), np.array([0,0,0])) for n in range(numBodies)]
        self.createOrbitPaths(bodies)
        for n in range(numBodies):
            offsetPositions = positions[n]
            origin = np.zeros([3, numSnaps])
            if(self.lockedBody == "" or self.lockedBody not in names): # Empty locked body => com
                com = np.zeros([3, numSnaps])
                masses = []
                for i in range(numBodies):
                    masses.append(next(b.mass for b in self.activeBodies if b.name == names[i]))
                    com = com + positions[i] * masses[i]
                com = com / sum(masses)
                origin = com
            else:
                origin = positions[names.index(self.lockedBody)]
            offsetPositions = offsetPositions - origin
            XnY = offsetPositions[0:2, :]
            Z = offsetPositions[2, :]
            self.orbitPaths[n].set_data(XnY)
            self.orbitPaths[n].set_3d_properties(Z)

    def animStart(self, pipeConnection):
        receiveThread = threading.Thread(target=self.getPipeData, name='receiver', args=(pipeConnection,1)) # The 2 should be replaced with something
        receiveThread.start()
        self.anim = animation.FuncAnimation(self.fig, self.animUpdate, 1, interval=1000, blit=False)
        #receiveThread.join() #! should I join receiveThread here, or is it automatically terminated?
