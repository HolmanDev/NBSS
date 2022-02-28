import numpy as np
import simulation as sim
from simulation import body, packet
import visualizer as vis
import libs.vectors as vec
import libs.physics as phy
import time
import pathlib
import threading
import multiprocessing as mp
#import yappi
import uiBackend as uiBack
import os
import json
import pandas as pd
from datetime import datetime

cachePath = "cache/"

activeSettings = uiBack.settings()
activeData = uiBack.data()
sim1 = sim.simulation()
sim1.minDist = 0#4.0 * phy.SR #10 * phy.au
sim1.collisionLogPath = "results/collisions/"
simulationProcesses = []

def awaitMessage(mainQueue, delay):
    while(True):
        if(mainQueue.get() == "collided"):
            uiBackend.collided()
            break
        time.sleep(delay)

def visualize():
    # Setup visualization
    sim1.setupOrbits(50, activeSettings.Kp, activeSettings.Ki, activeSettings.Kd, activeSettings.lowestErrorDamp, activeSettings.errorDampSharpness)
    sim1.positions = sim1.genPositions(activeSettings.snapsPerFrame, activeSettings.stepsPerSnap, activeSettings.timestep) # Setup simulation by generating the first position
    vis1 = vis.visualization()
    vis1.createOrbitPaths(sim1.bodies)
    vis1.setBoundries([0,0,0], activeSettings.zoom)
    vis1.lockedBody = activeSettings.focusBody
    vis1.markOrigin("+")
    vis1.uiBackend = uiBackend
    # Setup multiple processes
    mainQueue = mp.Queue()
    child_conn, parent_conn = mp.Pipe(duplex=False) # Setup pipe connection to communicate between the computation- and graphics-processes
    vis1.animStart(child_conn)
    # Setup simulation
    sim1.positionRebaseInterval = activeSettings.rebaseInterval
    sim1.pathLength = activeSettings.pathLength
    calcProcess = mp.Process(target=sim1.posGenLoop, \
        args=(activeSettings.snapsPerFrame, activeSettings.stepsPerSnap, activeSettings.timestep, \
        activeSettings.posSendDelay, activeSettings.startTime, mainQueue, parent_conn))
    calcProcess.start()
    parent_conn.close()
    simulationProcesses.append(calcProcess)
    awaitMessageThread = threading.Thread(target=awaitMessage, name='awaiter', args=(mainQueue, 1))
    awaitMessageThread.start()
    # Show the plot (until it is closed)
    vis.plt.show()
    #calcProcess.join()

# Make this into its own process too.
def dryRun(path):
    sim1.setupOrbits(50, activeSettings.Kp, activeSettings.Ki, activeSettings.Kd, activeSettings.lowestErrorDamp, activeSettings.errorDampSharpness)
    # Setup multiple processes
    mainQueue = mp.Queue()
    child_conn, parent_conn = mp.Pipe()
    # Setup simulation
    sim1.positionRebaseInterval = activeSettings.rebaseInterval
    calcProcess = mp.Process(target=sim1.packetGenLoop, \
        args=(activeSettings.snapsPerPacket, activeSettings.stepsPerSnap, activeSettings.timestep, \
        activeSettings.packSendDelay, activeSettings.startTime, mainQueue, parent_conn))
    calcProcess.start()
    simulationProcesses.append(calcProcess)
    # Logging
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    logProcess = mp.Process(target=log, args=(path, activeSettings.logInterval, mainQueue, child_conn))
    logProcess.start()
    simulationProcesses.append(logProcess)
    awaitMessageThread = threading.Thread(target=awaitMessage, name='awaiter', args=(mainQueue,1))
    awaitMessageThread.start()

def log(path, packetInterval, mainQueue, calcPipeConnection):
    lastTime = datetime.now()
    packet_index = 0
    msg = []
    while(True):
        now = datetime.now()
        if(now - lastTime).total_seconds() >= packetInterval or packet_index == 0:
            # Receive message from pipe and clear it (yes, pipes are like narrow queues)
            try:
                calcPipeConnection.poll(None) # Await packet
                while calcPipeConnection.poll():
                    try:
                        msg = calcPipeConnection.recv()
                        if msg[2] != "default": # Needed?
                            mainQueue.put('collided')
                            return
                    except:
                        break
            except:
                return         
            # Create log file
            filename = f"{path}packet_{packet_index}_{now.year}-{now.month}-{now.day}T{now.hour}-{now.minute}-{now.second}.txt"
            with open(filename, "w") as f:
                # Get packet data from the pipe message
                packet = msg[0]
                if packet is None:
                    continue
                # Log to file
                simTime = msg[1]
                f.write(str(simTime[0] + simTime[1] / phy.secondsInYear) + "\n")
                f.write("position,velocity\n")
                f.write(str(packet))
            # Move forward
            packet_index = packet_index + 1
            lastTime = datetime.now() # New time necessary due to recv()

def readLog(path):
    #sim1.positions = sim1.genPositions(snapsPerFrame, stepsPerSnap, timestep)
    vis1 = vis.visualization()
    vis1.setBoundries([0,0,0], activeSettings.zoom)
    vis1.lockedBody = activeSettings.focusBody
    vis1.markOrigin("+")
    vis1.readLog(path, activeSettings.showNth)
    vis.plt.show() # Split into another process?

def logAndVisualize(path):
    mainQueue = mp.Queue()
    calcChildConn, calcParentConn = mp.Pipe()
    calcProcess = mp.Process(target=sim1.packetGenLoop, \
        args=(activeSettings.snapsPerPacket, activeSettings.stepsPerSnap, activeSettings.timestep, \
            activeSettings.posSendDelay, activeSettings.startTime, calcParentConn))
    calcProcess.start()

    logProcess = mp.Process(target=log, args=(path, mainQueue, calcChildConn))
    logProcess.start()

    # Setup visualization
    sim1.positions = sim1.genPositions(activeSettings.snapsPerFrame, activeSettings.stepsPerSnap, activeSettings.timestep) # Setup simulation by generating the first position
    vis1 = vis.visualization()
    vis1.createOrbitPaths(sim1.bodies)
    vis1.setBoundries([0,0,0], activeSettings.zoom)
    vis1.lockedBody = activeSettings.focusBody
    vis1.markOrigin("+")
    # Setup multiple processes
    vis1.animStartFromLog(path)
    # Show the plot (until it is closed)
    vis.plt.show()

def startSimulation():
    if(activeSettings.dryRunEnabled):
        dryRun(activeSettings.logPath)
    if(activeSettings.visualizeEnabled):
        visualize()
    #logAndVisualize(logPath)

def stopSimulation():
    global simulationProcesses
    for process in simulationProcesses:
        process.terminate()
    simulationProcesses = []

def gatherData(fileNames):
    activeData.bodies = []
    for j in range(0, len(fileNames)):
        with open(fileNames[j], "r") as f:
            try:
                b = sim.body.parseJSON(f.read())
            except ValueError:
                continue
            activeData.bodies.append(b)
    sim1.bodies = activeData.bodies
    return activeData.bodies

def getSettingsCache():
    try:
        f = open(cachePath + "settings.txt", "r")
        try:
            global activeSettings
            activeSettings = uiBack.settings.parseAsJSON(f.read())
            f.close()
            print("Settings cache found")
        except:
            f.close()
            os.remove(cachePath + "settings.txt")
            print("Settings cache was found but couldn't be read, so it was deleted")
    except:
        print("No settings cache found")

def setSettingsCache():
    try:
        with open(cachePath + "settings.txt", "w") as outfile:
            activeSettings.toJSON(outfile)
        print("Saved settings")
    except:
        print("Failed to save settings")

def openEvent():
    print("Hello!")
    getSettingsCache()

def closeEvent(e): 
    setSettingsCache()
    print("Bye!")

if __name__ == '__main__':
    global uiBackend
    uiBackend = uiBack.uiBackend()
    openEvent()
    uiBackend.startFunc = startSimulation
    uiBackend.stopFunc = stopSimulation
    uiBackend.snapshotFunc = readLog
    uiBackend.browseFunc = gatherData
    uiBackend.closeWindowFunc = closeEvent
    uiBackend.setupUI(activeSettings)