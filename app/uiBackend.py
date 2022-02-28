import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from numpy.core.arrayprint import DatetimeFormat
import interface as intr
import os
import numpy as np
import pandas as pd
import json

from visualizer import visualization

class uiBackend:
    settings = -1
    ui = None

    startFunc = lambda: 0
    stopFunc = lambda: 0
    snapshotFunc = lambda: 0
    browseFunc = lambda: 0
    closeWindowFunc = lambda: 0

    # Simulation
    def timestepEditFinish(self):
        self.settings.timestep = float(self.ui.lnbxTimestep.text())

    def stepsPerSnapEditFinish(self):
        self.settings.stepsPerSnap = int(self.ui.lnbxStepsPerSnap.text())

    def snapsEditFinish(self):
        self.settings.snapsPerPacket = int(self.ui.lnbxSnaps.text())
        self.settings.snapsPerFrame = int(self.ui.lnbxSnaps.text())

    def logIntervalEditFinish(self):
        self.settings.logInterval = float(self.ui.lnbxLogInterval.text())

    def dateStartTimeEditFinish(self):
        self.settings.startTime = pd.to_datetime(self.ui.dateStartTime.date().toString()) # Right format?

    # Starting orbit optimization
    def KpEditFinish(self):
        self.settings.Kp = float(self.ui.lnbxKp.text())

    def KiEditFinish(self):
        self.settings.Ki = float(self.ui.lnbxKi.text())

    def KdEditFinish(self):
        self.settings.Kd = float(self.ui.lnbxKd.text())

    def lowestErrorDampEditFinish(self):
        self.settings.lowestErrorDamp = float(self.ui.lnbxLowestErrorDamp.text())

    def errorDampSharpnessEditFinish(self):
        self.settings.errorDampSharpness = float(self.ui.lnbxErrorDampSharpness.text())

    # View
    def focusEditFinish(self):
        self.settings.focusBody = self.ui.lnbxFocus.text()

    def zoomEditFinish(self):
        self.settings.zoom = float(self.ui.lnbxZoom.text())

    # General
    def updateLogState(self, state):
        self.settings.dryRunEnabled = state
        self.ui.lnbxLogWritePath.setEnabled(state)
        self.ui.lnbxLogReadPath.setEnabled(state)
        self.ui.lnbxNthPacket.setEnabled(state)
        self.ui.btnStart.setEnabled(state or self.ui.gbxVisualize.isChecked())

    def updateVisState(self, state):
        self.settings.visualizeEnabled = state
        self.ui.lnbxPathLength.setEnabled(state)
        self.ui.btnStart.setEnabled(state or self.ui.gbxLog.isChecked())

    def logWritePathEditFinish(self):
        self.settings.logPath = self.ui.lnbxLogWritePath.text()

    def logReadPathEditFinish(self):
        self.settings.logPath = self.ui.lnbxLogReadPath.text()

    def pathLengthEditFinish(self):
        self.settings.pathLength = int(self.ui.lnbxPathLength.text())

    def nthPacketEditFinish(self):
        self.settings.showNth = int(self.ui.lnbxNthPacket.text())

    # Advanced
    def rebaseIntervalEditFinish(self):
        self.settings.rebaseInterval = float(self.ui.lnbxRebaseInterval.text())

    # Start / Stop
    def startClicked(self):
        self.ui.btnStart.setText("Stop")
        self.ui.btnStart.clicked.connect(lambda: self.stopClicked())
        self.startFunc()

    def stopClicked(self):
        self.ui.btnStart.setText("Start")
        self.ui.btnStart.clicked.connect(lambda: self.startClicked())
        self.stopFunc()

    def snapshotClicked(self):
        self.snapshotFunc(self.settings.logPath)

    # Data
    def browseClicked(self, mainWindow):
        fname = QtWidgets.QFileDialog.getOpenFileNames(mainWindow, 'Open File', os.path.expanduser("~/Desktop"), 'JSON files (*.json)')
        self.ui.tbxData.clear()
        data = self.browseFunc(fname[0])
        self.ui.lnbxName.setText(str(fname[0]).replace("[", "").replace("]", ""))
        for body in data:
            self.ui.tbxData.append((str(body.name) + ":\n    mass: " + str(body.mass) + "\n    pos: " + str(body.pos) + "\n    vel: " + str(body.vel) + "\n")\
                .replace("[","").replace("]","").replace("array",""))

    def clearLogClicked(self):
        fileNames = [x for x in os.listdir(self.settings.logPath) if x.startswith("packet_")]
        for j in range(0, len(fileNames)):
            try:
                os.remove(self.settings.logPath + fileNames[j])
            except:
                continue
        
    def collided(self):
        self.ui.btnStart.setText("Start")
        self.ui.btnStart.clicked.connect(lambda: self.startClicked())

    def close(self, mainWindow, e):
        mainWindow.setFocus()
        self.closeWindowFunc(e)

    def setupUI(self, settings):
        self.settings = settings
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        self.ui = intr.Ui_MainWindow()
        self.ui.setupUi(MainWindow)

        # -SETTINGS-
        # Simulation
        self.ui.lnbxTimestep.setText(str(self.settings.timestep))
        self.ui.lnbxTimestep.editingFinished.connect(lambda: self.timestepEditFinish())
        self.ui.lnbxStepsPerSnap.setText(str(self.settings.stepsPerSnap))
        self.ui.lnbxStepsPerSnap.editingFinished.connect(lambda: self.stepsPerSnapEditFinish())
        self.ui.lnbxSnaps.setText(str(self.settings.snapsPerPacket))
        self.ui.lnbxSnaps.editingFinished.connect(lambda: self.snapsEditFinish())
        self.ui.lnbxLogInterval.setText(str(self.settings.logInterval))
        self.ui.lnbxLogInterval.editingFinished.connect(lambda: self.logIntervalEditFinish())
        self.ui.dateStartTime.setDate(QtCore.QDate(self.settings.startTime.year, self.settings.startTime.month, self.settings.startTime.day))
        self.ui.dateStartTime.editingFinished.connect(lambda: self.dateStartTimeEditFinish())
        # Starting orbit optimization
        self.ui.lnbxKp.setText(str(self.settings.Kp))
        self.ui.lnbxKp.editingFinished.connect(lambda: self.KpEditFinish())
        self.ui.lnbxKi.setText(str(self.settings.Ki))
        self.ui.lnbxKi.editingFinished.connect(lambda: self.KiEditFinish())
        self.ui.lnbxKd.setText(str(self.settings.Kd))
        self.ui.lnbxKd.editingFinished.connect(lambda: self.KdEditFinish())
        self.ui.lnbxLowestErrorDamp.setText(str(self.settings.lowestErrorDamp))
        self.ui.lnbxLowestErrorDamp.editingFinished.connect(lambda: self.lowestErrorDampEditFinish())
        self.ui.lnbxErrorDampSharpness.setText(str(self.settings.errorDampSharpness))
        self.ui.lnbxErrorDampSharpness.editingFinished.connect(lambda: self.errorDampSharpnessEditFinish()())

        # View
        self.ui.lnbxFocus.setText(self.settings.focusBody)
        self.ui.lnbxFocus.editingFinished.connect(lambda: self.focusEditFinish())
        self.ui.lnbxZoom.setText(str(self.settings.zoom))
        self.ui.lnbxZoom.editingFinished.connect(lambda: self.zoomEditFinish())

        # General
        self.ui.gbxVisualize.setChecked(self.settings.visualizeEnabled)
        self.ui.gbxVisualize.toggled.connect(lambda: self.updateVisState(self.ui.gbxVisualize.isChecked()))
        self.updateVisState(self.ui.gbxVisualize.isChecked())
        self.ui.lnbxPathLength.setText(str(self.settings.pathLength))
        self.ui.lnbxPathLength.editingFinished.connect(lambda: self.pathLengthEditFinish())
        self.ui.gbxLog.setChecked(self.settings.dryRunEnabled)
        self.ui.gbxLog.toggled.connect(lambda: self.updateLogState(self.ui.gbxLog.isChecked()))
        self.updateLogState(self.ui.gbxLog.isChecked())
        self.ui.lnbxLogWritePath.setText(self.settings.logPath)
        self.ui.lnbxLogWritePath.editingFinished.connect(lambda: self.logWritePathEditFinish())
        self.ui.lnbxLogReadPath.setText(self.settings.logPath)
        self.ui.lnbxLogReadPath.editingFinished.connect(lambda: self.logReadPathEditFinish())
        self.ui.lnbxNthPacket.setText(str(self.settings.showNth))
        self.ui.lnbxNthPacket.editingFinished.connect(lambda: self.nthPacketEditFinish())

        # Advanced
        self.ui.lnbxRebaseInterval.setText(str(self.settings.rebaseInterval))
        self.ui.lnbxRebaseInterval.editingFinished.connect(lambda: self.rebaseIntervalEditFinish())

        # -DATA-
        self.ui.btnBrowse.clicked.connect(lambda: self.browseClicked(MainWindow))

        # Start / Stop
        self.ui.btnStart.clicked.connect(lambda: self.startClicked())
        self.ui.btnSnapshot.clicked.connect(lambda: self.snapshotClicked())

        # Misc
        self.ui.btnClearLog.clicked.connect(lambda: self.clearLogClicked())

        MainWindow.closeEvent = lambda e: self.close(MainWindow, e)
        MainWindow.show()
        sys.exit(app.exec_())

class settings:
    # Simulation
    timestep = 3600.0 # simulated seconds per step
    stepsPerSnap = 50 # steps per snap
    snapsPerPacket = 10 # snaps per packet (for dry run)
    snapsPerFrame = 10 # snaps per frame (for animation)
    logInterval = 3 # seconds 
    startTime = pd.to_datetime('2000-01-01')
    Kp = 0.5
    Ki = 0
    Kd = 0.5
    lowestErrorDamp = 0.15
    errorDampSharpness = 0.025

    # View
    focusBody = ""
    zoom = 30.0

    # General
    visualizeEnabled = False
    pathLength = 10000 # Number of stored positions to be visualized
    dryRunEnabled = False
    logPath = "results/packets/" #"C:/Users/simon/Desktop/GA/Data/2 Cam/Sim1/" ##  #remove "packets/" and make that implicit
    showNth = 1 # Display the nth packet data

    # Advanced
    rebaseInterval = 100000.0
    posSendDelay = 2 # Delay between position deliveries
    packSendDelay = 2 # Delay between packet deliveries

    def toJSON(self, outfile):
        s = {
            "timestep": str(self.timestep),
            "stepsPerSnap": str(self.stepsPerSnap),
            "snapsPerPacket": str(self.snapsPerPacket),
            "snapsPerFrame": str(self.snapsPerFrame),
            #"posSendDelay": str(self.posSendDelay),
            #"packSendDelay": str(self.packSendDelay),
            "logInterval": str(self.logInterval),
            "startTime": str(self.startTime),
            "Kp": str(self.Kp),
            "Ki": str(self.Ki),
            "Kd": str(self.Kd),
            "lowestErrorDamp": str(self.lowestErrorDamp),
            "errorDampSharpness": str(self.errorDampSharpness),
            "focusBody": self.focusBody,
            "zoom": str(self.zoom),          
            "visualizeEnabled": str(self.visualizeEnabled),
            "pathLength": str(self.pathLength),
            "dryRunEnabled": str(self.dryRunEnabled),
            "showNth": str(self.showNth),
            "rebaseInterval": str(self.rebaseInterval)
        }
        return json.dump(s, outfile)

    def parseAsJSON(text):
        data = json.loads(text)
        s = settings()
        s.timestep = float(data['timestep'])
        s.stepsPerSnap = int(data['stepsPerSnap'])
        s.snapsPerPacket = int(data['snapsPerPacket'])
        s.snapsPerFrame = int(data['snapsPerFrame'])
        #s.posSendDelay = float(data['posSendDelay'])
        #s.packSendDelay = float(data['packSendDelay'])
        s.logInterval = float(data['logInterval'])
        s.startTime = pd.to_datetime(data['startTime'])
        s.Kp = float(data['Kp'])
        s.Ki = float(data['Ki'])
        s.Kd = float(data['Kd'])
        s.lowestErrorDamp = float(data['lowestErrorDamp'])
        s.errorDampSharpness = float(data['errorDampSharpness'])
        s.focusBody = data['focusBody']
        s.zoom = float(data['zoom'])
        s.visualizeEnabled = data['visualizeEnabled'] == 'True'
        s.pathLength = int(data['pathLength'])
        s.dryRunEnabled = data['dryRunEnabled'] == 'True'
        s.showNth = int(data['showNth'])
        s.rebaseInterval = float(data['rebaseInterval'])
        return s

class data:
    bodies = []