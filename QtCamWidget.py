#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 15:51:33 2020

@author: Roman Chernikov
"""

from __future__ import print_function
from functools import partial
from pyqtgraph.Qt import QtGui, QtCore

import numpy as np
import pyqtgraph as pg
import cv2
import ctypes
import json
import copy
import os
import datetime
try:
    from gi.repository import Aravis
    isGigE = True
except ImportError:
    isGigE = False
#import sys
try:
    import epics
    isEPICS = True
except ImportError:
    isEPICS = False
import time

try:
    qcd = cv2.QRCodeDetector()
    dAdM = qcd.detectAndDecodeMulti
    isQCD = True
except:
    isQCD = False

isQCD = False  # Disabled

class MyImageItem(pg.ImageItem):
    dotCatchSignal = QtCore.pyqtSignal(int, int)
    dotDragSignal = QtCore.pyqtSignal(int, int)
    dotReleaseSignal = QtCore.pyqtSignal(int, int)

#    def mouseClickEvent(self, event):
#        pass

    def mouseDragEvent(self, event):
        if event.isStart():
            self.dotCatchSignal.emit(event.pos().x(), event.pos().y())
        elif event.isFinish():
            self.dotReleaseSignal.emit(event.pos().x(), event.pos().y())
        else:
            self.dotDragSignal.emit(event.pos().x(), event.pos().y())

    def hoverEvent(self, event):
        if not event.isExit():
            event.acceptDrags(pg.QtCore.Qt.LeftButton)
            event.acceptClicks(pg.QtCore.Qt.LeftButton)


class SetupDialog(QtGui.QDialog):
    configChanged = QtCore.pyqtSignal(dict)
    applyConfig = QtCore.pyqtSignal(dict)
    setNewPath = QtCore.pyqtSignal(str)

    def __init__(self, config=None, rawFrame=None):
        super(SetupDialog, self).__init__()
        self.rawFrame = rawFrame
        self.cameraConfig = config

        self.tmpConfig = {}

        for field in ['gridEnabled', 'slitsEnabled', 'cmapEnabled',
                      'contrastEnhanced']:
            self.tmpConfig[field] = self.cameraConfig[field]

        self.frame_width = self.cameraConfig['frame_size'][0]
        self.frame_height = self.cameraConfig['frame_size'][1]
        layout = QtGui.QVBoxLayout()
        scrLayout = QtGui.QHBoxLayout()

        cmaps = []
        for key in cv2.__dict__.keys():
            keystr = str(key)
            if keystr.startswith("COLORMAP"):
                cmaps.append(keystr.split("_")[1])

        rawView = pg.GraphicsView(self)
        rawvb = pg.ViewBox()
        rawView.setCentralItem(rawvb)
        rawvb.setAspectLocked(True)
        self.imgRaw = MyImageItem(axisOrder='row-major')
        self.imgRaw.dotCatchSignal.connect(self.catch_dot)
        self.imgRaw.dotDragSignal.connect(self.drag_dot)
        self.imgRaw.dotReleaseSignal.connect(self.release_dot)
#        self.imgRaw = pg.ImageItem(axisOrder='row-major')
        rawvb.addItem(self.imgRaw)
        rawvb.invertY()

        origWidth, origHeight = self.cameraConfig['original_resolution']

        self.rawWidth = 600
        self.rawHeight = int(float(self.rawWidth)*(float(origHeight)/float(origWidth)))

        self.gView2 = pg.GraphicsView(self)
        self.vb2 = pg.ViewBox()
        self.gView2.setCentralItem(self.vb2)
        self.vb2.setAspectLocked(True)
        self.imgProcessed = pg.ImageItem(axisOrder='row-major')

        self.vb2.addItem(self.imgProcessed)
        self.vb2.invertY()

        rawView.setFixedSize(self.rawWidth, self.rawHeight)
        rawvb.setRange(QtCore.QRectF(0, 0, self.rawWidth, self.rawHeight),
                    padding=0)

        self.gView2.setFixedSize(self.frame_width, self.frame_height)
        self.vb2.setRange(QtCore.QRectF(0, 0, self.frame_width, self.frame_height),
                    padding=0)

        scrLayout.addWidget(rawView)
        scrLayout.addWidget(self.gView2)


        self.scales = [float(origWidth)/float(self.rawWidth),
                       float(origHeight)/float(self.rawHeight)]
        if self.cameraConfig['warp'] is not None:
            self.dots = np.int32(
                    np.array(self.cameraConfig['warp'])/np.array(self.scales))
        else:
            self.dots = [(10, 10), (self.rawWidth-10, 10),
                         (self.rawWidth-10, self.rawHeight-10),
                         (10, self.rawHeight-10)]
        self.lines = [0]*4
        self.recalculate_lines()
        self.movingDot = None

        panels = QtGui.QWidget()
        pLayout = QtGui.QGridLayout()

        panelGeneral = QtGui.QGroupBox("General")
        pgLayout = QtGui.QGridLayout(panelGeneral)
        pgLayout.addWidget(QtGui.QLabel("Name"), 0, 0, 1, 1)
        nameQLE = QtGui.QLineEdit(str(self.cameraConfig['name']))
        nameQLE.settingsDictKey = 'name'
        nameQLE.editingFinished.connect(self.update_config)
        pgLayout.addWidget(nameQLE, 0, 1, 1, -1)
        pgLayout.addWidget(QtGui.QLabel("Address"), 1, 0, 1, 1)
        addrQLE = QtGui.QLineEdit(str(self.cameraConfig['address']))
        addrQLE.settingsDictKey = 'address'
        addrQLE.editingFinished.connect(self.update_config)
        pgLayout.addWidget(addrQLE, 1, 1, 1, -1)
        pgLayout.addWidget(QtGui.QLabel("Type"), 2, 0, 1, 1)

        typeCombo = QtGui.QComboBox()
        typeCombo.addItems(["IP", "USB"])
        if isGigE:
            typeCombo.addItem("GigE")
        typeCombo.settingsDictKey = 'type'
        typeCombo.setCurrentText(self.cameraConfig['type'])
        typeCombo.currentTextChanged.connect(self.update_config)
        pgLayout.addWidget(typeCombo, 2, 1, 1, -1)

        pgLayout.addWidget(QtGui.QLabel("Save path"), 3, 0, 1, 1)
        pathQLE = QtGui.QLineEdit(str(self.cameraConfig['savedir']))
        pathQLE.settingsDictKey = 'savedir'
        pathQLE.editingFinished.connect(self.update_config)
        self.setNewPath.connect(pathQLE.setText)
        pgLayout.addWidget(pathQLE, 3, 1, 1, 2)

        fdButton = QtGui.QPushButton("Select")
        fdButton.clicked.connect(self.update_savepath)
        fdButton.setAutoDefault(False)
        pgLayout.addWidget(fdButton, 3, 3)

        panelColor = QtGui.QGroupBox("Image")
        pcLayout = QtGui.QGridLayout(panelColor)

        pcLayout.addWidget(QtGui.QLabel("Colormap"), 0, 0)
        cmSelector = QtGui.QComboBox()
        cmSelector.settingsDictKey = 'colormap'
        cmSelector.addItems(cmaps)
        cmSelector.setCurrentText(self.cameraConfig['colormap'])
        cmSelector.currentTextChanged.connect(self.update_config)
        pcLayout.addWidget(cmSelector, 0, 1, 1, 3)

        cmCB = QtGui.QCheckBox()
        cmCB.setChecked(self.cameraConfig['cmapEnabled'])
        cmCB.settingsDictKey = 'cmapEnabled'
        cmCB.stateChanged.connect(self.update_config)
        pcLayout.addWidget(cmCB, 0, 4)

        pcLayout.addWidget(QtGui.QLabel("Brightness"), 1, 0)
        cbSlider = QtGui.QSlider()
        cbSlider.settingsDictKey = 'brightness'
        cbSlider.setValue(self.cameraConfig['brightness'])
        cbSlider.setOrientation(QtCore.Qt.Horizontal)
        cbSlider.setRange(-32, 32)
        cbSlider.valueChanged.connect(self.update_config)
        pcLayout.addWidget(cbSlider, 1, 1, 1, 3)

        pcLayout.addWidget(QtGui.QLabel("Contrast"), 2, 0)
        ccSlider = QtGui.QSlider()
        ccSlider.settingsDictKey = 'contrast'
        ccSlider.setValue(self.cameraConfig['contrast'])
        ccSlider.setOrientation(QtCore.Qt.Horizontal)
        ccSlider.setRange(-32, 32)
        ccSlider.valueChanged.connect(self.update_config)
        pcLayout.addWidget(ccSlider, 2, 1, 1, 3)

        brcrCB = QtGui.QCheckBox()
        brcrCB.setChecked(self.cameraConfig['contrastEnhanced'])
        brcrCB.settingsDictKey = 'contrastEnhanced'
        brcrCB.stateChanged.connect(self.update_config)
        pcLayout.addWidget(brcrCB, 1, 4, 2, 1)

        pcLayout.addWidget(QtGui.QLabel("Resolution"), 3, 0, 1, 1)
        pcLayout.addWidget(QtGui.QLabel("x"), 3, 2, 1, 1)

        for ipos, label in enumerate(["H", "V"]):
            frameResolutionQLE = QtGui.QLineEdit(str(self.cameraConfig['frame_size'][ipos]))
            frameResolutionQLE.settingsDictKey = ('frame_size', ipos)
            frameResolutionQLE.editingFinished.connect(self.update_config)
            pcLayout.addWidget(frameResolutionQLE, 3, ipos*2+1, 1, 1)

        panelGrid = QtGui.QGroupBox("Grid")
        panelGrid.setCheckable(True)
        panelGrid.setChecked(self.cameraConfig['gridEnabled'])
        panelGrid.settingsDictKey = 'gridEnabled'
        panelGrid.toggled.connect(self.update_config)
        pgrLayout = QtGui.QGridLayout(panelGrid)

        for ipos, label in enumerate(["Xmin", "Xmax", "Ymin", "Ymax"]):
            rowPos = ipos // 2
            colPos = ipos % 2
            pgrLayout.addWidget(QtGui.QLabel(label), rowPos, colPos*2)
            limQLE = QtGui.QLineEdit(str(self.cameraConfig['gridLimits'][ipos]))
            limQLE.settingsDictKey = ('gridLimits', ipos)
            limQLE.editingFinished.connect(self.update_config)
            pgrLayout.addWidget(limQLE, rowPos, colPos*2+1)

        for ipos, label in enumerate(["Hlines", "Vlines"]):
            pgrLayout.addWidget(QtGui.QLabel(label), 2, ipos*2)
            grdDensityQLE = QtGui.QLineEdit(str(self.cameraConfig['gridDensity'][ipos]))
            grdDensityQLE.settingsDictKey = ('gridDensity', ipos)
            grdDensityQLE.editingFinished.connect(self.update_config)
            pgrLayout.addWidget(grdDensityQLE, 2, ipos*2+1)
#
#        for ipos, label in enumerate(["Xmax", "Ymax", "Vlines", "Color"]):
#            pgrLayout.addWidget(QtGui.QLabel(label), ipos, 2)
#            pgrLayout.addWidget(QtGui.QLineEdit(""), ipos, 3)

        panelSlits = QtGui.QGroupBox("Slits")
        panelSlits.setCheckable(True)
        panelSlits.setChecked(self.cameraConfig['slitsEnabled'])
        panelSlits.settingsDictKey = 'slitsEnabled'
        panelSlits.toggled.connect(self.update_config)
        psLayout = QtGui.QGridLayout(panelSlits)

        for ipos, label in enumerate(["Left", "Right", "Bottom", "Top"]):
            psLayout.addWidget(QtGui.QLabel(label), ipos, 0)
            if isinstance(self.cameraConfig['slitPVs'], list):
                strPV = str(self.cameraConfig['slitPVs'][ipos])
            else:
                strPV = ""
            slitQLE = QtGui.QLineEdit(strPV)
            slitQLE.settingsDictKey = ('slitPVs', ipos)
            slitQLE.editingFinished.connect(self.update_config)
            psLayout.addWidget(slitQLE, ipos, 1)

        pLayout.addWidget(panelGeneral, 0, 0)
        pLayout.addWidget(panelColor, 0, 1)
        pLayout.addWidget(panelGrid, 1, 0)
        pLayout.addWidget(panelSlits, 1, 1)
        panels.setLayout(pLayout)

        bBox = QtGui.QHBoxLayout()

        okButton = QtGui.QPushButton('OK')
        okButton.clicked.connect(self.accept)
        okButton.setAutoDefault(False)

        applyButton = QtGui.QPushButton('Apply')
        applyButton.clicked.connect(self.apply)
        applyButton.setAutoDefault(False)

        cancelButton = QtGui.QPushButton('Cancel')
        cancelButton.clicked.connect(self.reject)
        cancelButton.setAutoDefault(False)

        bBox.addStretch()
        bBox.addWidget(okButton)
        bBox.addWidget(applyButton)
        bBox.addWidget(cancelButton)

        layout.addLayout(scrLayout)
        layout.addWidget(panels)
        layout.addLayout(bBox)

        self.setLayout(layout)

        self.setWindowTitle('Setup')

        self.update_raw(self.rawFrame)

        self.procThread = QtCore.QThread(self)
        self.iProc = ImageProcessor(config=copy.copy(self.cameraConfig),
                                    rawFrame=copy.copy(self.rawFrame))
        self.configChanged.connect(self.iProc.change_config)

        self.iProc.processedFrame.connect(self.update_processed)
        self.iProc.moveToThread(self.procThread)
        self.procThread.start()
        self.iProc.convert_frame(self.rawFrame)

    def accept(self):
        self.apply()
        self.procThread.terminate()
        super().accept()

    def closeEvent(self, event):
        self.procThread.terminate()
        event.accept()

    def reject(self):
        self.procThread.terminate()
        super().reject()

    def apply(self):
        for key, val in self.tmpConfig.items():
            self.cameraConfig[key] = val
        self.applyConfig.emit(copy.copy(self.cameraConfig))

    def recalculate_lines(self):
        for i, dot in enumerate(self.dots):
            np = i+1 if i<3 else 0
            x0 = dot[0]
            y0 = dot[1]
            dx = self.dots[np][0] - x0
            dy = self.dots[np][1] - y0
            p = [0]*2

            if dx == 0:
                p[0] = (int(x0), int(0))
                p[1] = (int(x0), int(self.rawHeight))

            else:
                a = float(dy)/float(dx)
                p[0] = (int(0), int(-x0*a + y0))
                p[1] = (int(self.rawWidth), int((self.rawWidth-x0)*a + y0))

                for point in p:
                    if point[1] < 0:
                        point = (int(-y0/a + x0), 0)
                    elif point[1] > self.rawHeight:
                        point = (int((point[1] - y0)/a + x0), int(self.rawHeight))

            self.lines[i] = p

    def draw_warp_grid(self, data):
        for idot, dot in enumerate(self.dots):
            data = cv2.line(data, self.lines[idot][0], self.lines[idot][1],
                            (0, 255, 0), 1)
            data = cv2.circle(data, (int(dot[0]), int(dot[1])), 10, (0, 255, 0), 2)
        return data

    def catch_dot(self, x, y):
        for idot, dot in enumerate(self.dots):
            dist = np.sqrt((dot[0]-x)**2 + (dot[1]-y)**2)
            if dist < 20:
                self.movingDot = idot
                break

    def drag_dot(self, x, y):
        if self.movingDot is not None:
            self.dots[self.movingDot] = (x, y)
            self.cameraConfig['warp'] = np.int32(
                    np.array(self.dots)*np.array(self.scales)).tolist()
            self.configChanged.emit(copy.copy(self.cameraConfig))
            self.recalculate_lines()
            self.update_raw(self.rawFrame)

    def release_dot(self, x, y):
        self.movingDot = None

    def update_savepath(self):
        fpDialog = QtGui.QFileDialog()
        fpDialog.setFileMode(QtGui.QFileDialog.Directory)
        defaultDir = os.path.abspath(self.cameraConfig['savedir'])
        if not os.path.exists(defaultDir):
            defaultDir = os.path.abspath('.')
        fpDialog.setDirectory(defaultDir)
        if fpDialog.exec_():
            newDir = fpDialog.directory().absolutePath()
            print(str(newDir))
            self.setNewPath.emit(str(newDir))
            self.cameraConfig['savedir'] = str(newDir)

    def update_raw(self, dataIn):
        if dataIn is None:
            return
        data = np.copy(dataIn)
        if data.shape[1] != self.rawWidth or data.shape[0] != self.rawHeight:
            data = cv2.resize(data, (self.rawWidth, self.rawHeight))
        if len(data.shape) < 3:
            data = cv2.applyColorMap(data, cv2.COLORMAP_BONE)
        data = self.draw_warp_grid(data)
        self.imgRaw.setImage(cv2.cvtColor(data, cv2.COLOR_BGR2RGB),
                          autoLevels=False, autoDownsample=False)

    def update_processed(self, data):
        if data is None:
            return
        self.imgProcessed.setImage(cv2.cvtColor(data, cv2.COLOR_BGR2RGB),
                          autoLevels=False, autoDownsample=True) #False)

    def update_config(self, data=None):
        sender = self.sender()
        config_field = sender.settingsDictKey

        if data is None:  # QLE
            data = str(sender.text())

        if config_field in ['cmapEnabled', 'contrastEnhanced']:
            data = bool(data)

        if isinstance(config_field, tuple):
            try:
                if config_field[0] in ['gridLimits']:
                    data = float(data)
                elif config_field[0] not in ['slitPVs']:
                    data = int(data)
            except ValueError:
                pass  # it's ok to be a string
            self.cameraConfig[config_field[0]][config_field[1]] = data
            if config_field[0] == 'frame_size':
                self.update_proc_size()
        else:
            try:
                if config_field in ['address', 'contrast', 'brightness']:
                    data = int(data)
            except ValueError:
                pass  # it's ok to be a string
            self.cameraConfig[config_field] = data
        self.configChanged.emit(copy.copy(self.cameraConfig))

    def update_proc_size(self):
        self.frame_width, self.frame_height = self.cameraConfig['frame_size']
        self.gView2.setFixedSize(self.frame_width, self.frame_height)
        self.vb2.setRange(QtCore.QRectF(0, 0, self.frame_width, self.frame_height),
                            padding=0)
#        self.adjustSize()

class CameraListener(QtCore.QObject):
    newFrame = QtCore.pyqtSignal(np.ndarray)
#    startTimerSignal = QtCore.pyqtSignal()
#    stopTimerSignal = QtCore.pyqtSignal()

    def __init__(self, config=None):
        super(CameraListener, self).__init__()
        self.cameraConfig = config
        self.defaultImage = None
#        self.timer = QtCore.QTimer()
#        self.timer.timeout.connect(self.killCamera)
#        self.startTimerSignal.connect(self.startTimer)
#        self.stopTimerSignal.connect(self.stopTimer)

#    def startTimer(self):
#        self.timer.start(5000)
#        print("Timer started")

#    def stopTimer(self):
#        self.timer.stop()
#        print("Timer stopped")
        
#    def killCamera(self):
#        print("reinitializing camera")
#        self.stopTimerSignal.emit()
#        self.init_camera()
#        self.stream.push_buffer(Aravis.Buffer.new_allocate(0))
#        raise Exception("Error creating buffer")
#        self.stream = None

#        self.init_camera()

    def init_camera(self):
        camResolution = self.cameraConfig['original_resolution']
        frame_rate = self.cameraConfig['frame_rate']
        exp_time = 1e6/float(frame_rate)

        self.stream = None
        self.listening = False

        for retry in range(10):
            try:
                initOK = False
                if self.cameraConfig['type'] == 'GigE' and isGigE:
                    # GiGE camera interface init. Not required for compressed IP cam stream
                    Aravis.enable_interface ("Fake")

                    # GiGE camera initialization, ten attempts usually enough
                    for j in range(10):
                        camera = Aravis.Camera.new(self.cameraConfig['address'])
                        camera.set_region(0, 0, camResolution[0], camResolution[1])
                        camera.set_frame_rate(frame_rate)
                        camera.set_exposure_time(exp_time)  # us
                        camera.set_gain(0.1)
                        camera.set_pixel_format(Aravis.PIXEL_FORMAT_MONO_8)
                        payload = camera.get_payload()
                        try:
                            self.stream = camera.create_stream(None, None)

                            for i in range(0,10):
                                self.stream.push_buffer(Aravis.Buffer.new_allocate(payload))
                            break
                        except AttributeError as err:
                            print(err)
                            print("Can't initialize camera. Retry in 1 s.")
                            time.sleep(1)
                    time.sleep(1)
                    camera.start_acquisition()
                    print("Camera initialized. Starting stream.")
                    initOK = True
                else:  # IP/USB camera
                    self.stream = cv2.VideoCapture(self.cameraConfig['address'])
                    rawWidth = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
                    rawHeight = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    if rawWidth > 0 and rawHeight > 0:
                        self.cameraConfig['original_resolution'] = [rawWidth,
                                         rawHeight]
                        initOK = True
                self.listening = initOK
                break
            except:
                print('Initialization failed. Retry in 1 s.')
                self.stream = None
                time.sleep(1)

    def convert(self, buf):
        "This function is GiGE specific. Converting Aravis buffer to numpy array."
        if not buf:
            return None
        pixel_format = buf.get_image_pixel_format()
        bits_per_pixel = pixel_format >> 16 & 0xff
        if bits_per_pixel == 8:
            INTP = ctypes.POINTER(ctypes.c_uint8)
        else:
            INTP = ctypes.POINTER(ctypes.c_uint16)
        addr = buf.get_data()
        ptr = ctypes.cast(addr, INTP)
        im = np.ctypeslib.as_array(ptr, (buf.get_image_height(),
                                         buf.get_image_width()))
        im = im.copy()
        return im

    def send_default(self):
        if self.defaultImage is not None:
            self.newFrame.emit(
                    cv2.resize(self.defaultImage,
                               (int(self.cameraConfig['frame_size'][0]),
                                int(self.cameraConfig['frame_size'][1]))))

    def listen_to_camera(self):  #, stream):
        """Main worker process: receiving the frames from camera, transforming
        and sending to the pyqtgraph ViewBox."""

#        self.stream = stream
        error_counter = 0
        while True:
#            if error_counter > 0:
#                print("error", error_counter, self.listening)
            if error_counter > 10:
                print("killing stream")
                if self.cameraConfig['type'] != 'GigE':
                    self.stream.release()
                self.stream = None
                self.listening = False
                self.send_default()
                time.sleep(5)
                error_counter = 0
                print("initializing camera")
                self.init_camera()

            if self.listening:
                try:
                    if self.cameraConfig['type'] == 'GigE' and isGigE:
#                        self.startTimerSignal.emit()
                        buffer = self.stream.timeout_pop_buffer(3e6)  # Image frame in Aravis format
#                        self.stopTimerSignal.emit()
                    else:
                        buffer, buffarray = self.stream.read()

                    if buffer:
                        # Convert frame to numpy array and apply perspective correction

                        if self.cameraConfig['type'] == 'GigE' and isGigE:
                            buffarray = self.convert(buffer)
                        # Send the raw frame
                        self.newFrame.emit(buffarray)
                        # Refresh Aravis buffer
                        if self.cameraConfig['type'] == 'GigE' and isGigE:
                            self.stream.push_buffer(buffer)
                        error_counter = 0
                    else:
                        self.send_default()
                        print("bad buffer")
                        error_counter += 1
                        time.sleep(1.)
                except Exception as e:
                    print(e)
                    self.send_default()
                    print("error while streaming")
                    error_counter += 1
                    time.sleep(1.)
            else:
                self.send_default()
                error_counter += 1
                print("not listening")
                time.sleep(1.)


class ImageProcessor(QtCore.QObject):
    processedFrame = QtCore.pyqtSignal(np.ndarray)
    newHist = QtCore.pyqtSignal(list)

    def __init__(self, config=None, rawFrame=None):
        super(ImageProcessor, self).__init__()
        self.timestamp = time.time()
        self.rawFrame = rawFrame
#        self.cameraConfig = config
        self.change_config(config)


    def change_config(self, new_config):
#        if new_config['gamma'] != self.cameraConfig['gamma']:
#            gamma0 = float(new_config['gamma'])
#            gamma = gamma0**np.sign(gamma0)
#            self.lookUpTable = np.empty((1,256), np.uint8)
#            for i in range(256):
#                self.lookUpTable[0,i] = np.clip(np.power(i / 255.0, gamma) * 255.0, 0, 255)
        self.cameraConfig = new_config
        self.frame_width, self.frame_height = self.cameraConfig['frame_size']

        if self.cameraConfig['warp'] is not None:
            transPoints0 = np.float32(self.cameraConfig['warp'])
            transPoints1 = np.float32([[0, 0],
                                       [self.frame_width, 0],
                                       [self.frame_width, self.frame_height],
                                       [0, self.frame_height]])

            self.warpMatr = cv2.getPerspectiveTransform(transPoints0,
                                                        transPoints1)
        else:
            self.warpMatr = None
        if self.rawFrame is not None:
            self.convert_frame(self.rawFrame)

    def draw_grid(self, img, width, height, xticks, yticks, thickness, color):
        for xn in np.linspace(0, width-1, xticks):
            cv2.line(img, (int(xn), int(0)), (int(xn), int(height)), color,
                     int(thickness))
        for yn in np.linspace(0, height-1, yticks):
            cv2.line(img, (int(0), int(yn)), (int(width), int(yn)), color,
                     int(thickness))
        cv2.line(img, (int(width/2), int(0)), (int(width/2), int(height)), color,
                 int(thickness+1))
        cv2.line(img, (int(0), int(height/2)), (int(width), int(height/2)), color,
                 int(thickness+1))
        return img

    def convert_frame(self, buffarray):
        if buffarray is None:
            return
        if self.cameraConfig['warp'] is not None:
            buffarray = cv2.warpPerspective(
                buffarray, self.warpMatr,
                (int(self.frame_width), int(self.frame_height)))
        elif buffarray.shape[0] != self.frame_height:
            buffarray = cv2.resize(buffarray, (int(self.frame_width),
                                               int(self.frame_height)))
        locTime = time.time()
        if isQCD and locTime - self.timestamp > 5:
            self.timestamp = locTime
            retval, decoded_info, points, straight_qrcode = qcd.detectAndDecodeMulti(buffarray)
           
            #retval = True if QRCode detected, decoded_info = info on the QR code
            if retval:
                print('The filter is ', decoded_info)
            else:
                print('Error reading the QR Code')

        if 'contrastEnhanced' in self.cameraConfig:
            if self.cameraConfig['contrastEnhanced']:  # Enhance contrast
#                buffarray = cv2.LUT(buffarray, self.lookUpTable)
                gain = (float(self.cameraConfig['contrast'])+33.)/32.
                bias = float(self.cameraConfig['brightness'])*4
                buffarray = cv2.convertScaleAbs(buffarray, alpha=gain, beta=bias)

        if 'histEnabled' in self.cameraConfig:
            if self.cameraConfig['histEnabled']:
                if len(buffarray.shape) > 2:
                    buffarrayGr = cv2.cvtColor(buffarray, cv2.COLOR_BGR2GRAY)
                else:
                    buffarrayGr = buffarray
                vHist = np.sum(buffarrayGr, axis=0)
                hHist = np.sum(buffarrayGr, axis=1)
                self.newHist.emit([vHist, hHist])

        if 'cmapEnabled' in self.cameraConfig:
            cmap = cv2.__dict__['COLORMAP_{}'.format(self.cameraConfig['colormap'])]
            if self.cameraConfig['cmapEnabled']:  # Apply colormap
                buffarray = cv2.applyColorMap(buffarray, cmap)
#            elif len(buffarray.shape) < 3:
#                buffarray = cv2.applyColorMap(buffarray, cv2.COLORMAP_BONE)

#        if 'slitsEnabled' in self.cameraConfig:
#            if self.cameraConfig['slitsEnabled']:  # Show slits
#                buffarray = self.draw_slits(
#                    buffarray, self.frame_width, self.frame_height)

        if 'gridEnabled' in self.cameraConfig:
            if self.cameraConfig['gridEnabled']:  # Show grid
                buffarray = self.draw_grid(
                    buffarray, self.frame_width, self.frame_height,
                    self.cameraConfig['gridDensity'][0],
                    self.cameraConfig['gridDensity'][1],
                    1, (0, 220, 0))
        self.processedFrame.emit(buffarray)


class VideoCamStreamer(QtGui.QWidget):
    procFrame = QtCore.pyqtSignal(np.ndarray)
    reInit = QtCore.pyqtSignal()
#    configChanged = QtCore.pyqtSignal(dict)

    def __init__(self, configFile=None, config=None, appConfig=None):
        super(VideoCamStreamer, self).__init__()

        camConfigDefault = {
                            'name': 'CameraName',
                            'address': None,
                            'type': "IP",  # "USB", "GigE"
                            'frame_size': [640, 480],
                            'original_resolution': [640, 480],
                            'frame_rate': 5,  # fps
                            'warp': None,
                            'gridEnabled': False,
                            'gridLimits': [-1, 1, -1, 1],  # left, right, bottom, top
                            'gridDensity': [15, 11],
                            'histEnabled': False,
                            'slitsEnabled': False,
                            'slitPVs': None,
                            'cmapEnabled': True,
                            'colormap': 'JET',
                            'contrastEnhanced': False,
                            'contrast': 0,
                            'brightness': 0,
                            'savedir': os.path.abspath('.')
                            }

        appConfigDefault = {'buttons': ['grid', 'slits', 'contrast',
                                        'colormap', 'hist'],
                           'iconSize': (48, 48)}

        self.cameraConfig = config

        if configFile is not None:
            try:
                self.configFile = configFile
                with open(self.configFile, "r") as read_file:
                    self.cameraConfig = json.load(read_file)
            except:
                print("File load failed")

        try:
            emptyImage = cv2.imread('PM5544.png', cv2.IMREAD_UNCHANGED)
        except:
            emptyImage = None

        if self.cameraConfig is None:
            self.cameraConfig = camConfigDefault

        if 'configFile' in self.cameraConfig:
            self.configFile = self.cameraConfig['configFile']

        self.appConfig = appConfig if appConfig is not None else appConfigDefault

        self.currentImage = None
        self.rawFrame = None

        layout = QtGui.QHBoxLayout()
        self.iconsDir = ('.')
        self.gView = pg.GraphicsView(self)

        self.vb = pg.ViewBox()
        self.gView.setCentralItem(self.vb)
        self.vb.setAspectLocked(True)
        self.img = pg.ImageItem(axisOrder='row-major')

        self.vb.addItem(self.img)
        self.vb.invertY()

        for key, val in camConfigDefault.items():
            if key not in self.cameraConfig:
                self.cameraConfig[key] = val

        self.init_toolbar()
        self.init_slits()
        layout.addWidget(self.toolBar)

        histLayout = QtGui.QGridLayout()
        vHistWidget = pg.PlotWidget()
        hHistWidget = pg.PlotWidget()

        self.vHist = vHistWidget.plot(pen=pg.mkPen((0, 0, 0), width=1),
                                      fillLevel=0,
                                      brush=pg.mkBrush(0, 0, 212, 32))
        self.hHist = hHistWidget.plot(pen=pg.mkPen((0, 0, 0), width=1),
                                      fillLevel=0,
                                      brush=pg.mkBrush(0, 0, 212, 32))
        self.vHist.rotate(-90)
        vHistWidget.hideAxis('bottom')
        hHistWidget.hideAxis('left')
#        hHistWidget.hideAxis('bottom')
#        vHistWidget.hideAxis('left')
#        vHistWidget.invertY()
        vHistWidget.setBackground("w")
        hHistWidget.setBackground("w")
        self.hHistWidget = hHistWidget
        self.vHistWidget = vHistWidget

        histLayout.addWidget(self.gView, 0, 0)
        histLayout.addWidget(vHistWidget, 0, 1)
        histLayout.addWidget(hHistWidget, 1, 0)
        layout.addLayout(histLayout)
        self.setLayout(layout)

        self.toolBarHeight = self.appConfig['iconSize'][0]+4
        self.histogramHeight = 150

#        self.toolBar.setFixedHeight(self.toolBarHeight)

#        self.newFrame.connect(self.update_frame)
        self.gridLimits = self.cameraConfig['gridLimits']
        self.gridDensity = self.cameraConfig['gridDensity']

        self.update_geometry()
#        self.init_camera()

        self.toggle_hist(self.cameraConfig['histEnabled'])
        if 'hist' not in self.appConfig['buttons']:
            self.hHistWidget.setVisible(False)
            self.vHistWidget.setVisible(False)
            self.histogramHeight = 0

        self.setWindowTitle(self.cameraConfig['name'])
        self.listenerThread = QtCore.QThread()
        self.listener = CameraListener(self.cameraConfig)
        self.reInit.connect(self.listener.init_camera)
        self.listener.newFrame.connect(self.process_frame)
        self.listener.moveToThread(self.listenerThread)
#        startListening = partial(self.listener.listen_to_cam, self.stream)
        self.listenerThread.started.connect(self.listener.listen_to_camera)
        self.listener.defaultImage = emptyImage

        self.processThread = QtCore.QThread()
        self.procObj = ImageProcessor(config=self.cameraConfig)
        self.listener.newFrame.connect(self.procObj.convert_frame)
        self.procObj.processedFrame.connect(self.update_frame)
        self.procObj.newHist.connect(self.update_hist)
        self.procObj.moveToThread(self.processThread)
        self.processThread.start()
        self.listenerThread.start()
        self.reInit.emit()

    def toggle_hist(self, status):
        self.cameraConfig['histEnabled'] = status
        if status:
            self.setFixedSize(self.frame_width+40+self.toolBarHeight+self.histogramHeight,
                              self.frame_height+25+self.histogramHeight)
            self.hHistWidget.setVisible(True)
            self.vHistWidget.setVisible(True)

        else:
            self.setFixedSize(self.frame_width+40+self.toolBarHeight,
                              self.frame_height+25)
            self.hHistWidget.setVisible(False)
            self.vHistWidget.setVisible(False)

    def toggle_grid(self, status):
        self.cameraConfig['gridEnabled'] = status

    def toggle_slits(self, status):
        self.cameraConfig['slitsEnabled'] = status

    def toggle_cmap(self, status):
        self.cameraConfig['cmapEnabled'] = status
#        self.procObj.change_config(copy.copy(self.))

    def toggle_contrast(self, status):
        self.cameraConfig['contrastEnhanced'] = status

    def init_slits(self):
        if self.cameraConfig['slitPVs'] is not None and isEPICS:
            self.slits = [epics.PV(i) for i in self.cameraConfig['slitPVs']]
        else:
            self.slits = None

    def init_toolbar(self):
        self.toolBar = QtGui.QToolBar('Action buttons')
        self.toolBar.setOrientation(QtCore.Qt.Vertical)
        self.toolBar.setIconSize(QtCore.QSize(*self.appConfig['iconSize']))


        self.gridAction = QtGui.QAction(
            QtGui.QIcon(os.path.join(self.iconsDir, 'grid.png')),
            'Show Grid', self)
        self.gridAction.setShortcut('Alt+G')
        self.gridAction.setIconText('Show Grid')
        self.gridAction.triggered.connect(self.toggle_grid)
        self.gridAction.setCheckable(True)
        if 'grid' in self.appConfig['buttons']:
            if 'gridEnabled' in self.cameraConfig:
                self.gridAction.setChecked(self.cameraConfig['gridEnabled'])
            self.toolBar.addAction(self.gridAction)
            self.toolBar.addSeparator()
        else:
            self.gridAction.setChecked(False)


        if self.cameraConfig['slitPVs'] is not None and isEPICS:
            self.slitsAction = QtGui.QAction(
                QtGui.QIcon(os.path.join(self.iconsDir, 'slits.png')),
                'Show Slits', self)
            self.slitsAction.setShortcut('Alt+S')
            self.slitsAction.setIconText('Show Slits')
            self.slitsAction.triggered.connect(self.toggle_slits)
            self.slitsAction.setCheckable(True)
            if 'slitsEnabled' in self.cameraConfig:
                self.slitsAction.setChecked(self.cameraConfig['slitsEnabled'])
            if 'slits' in self.appConfig['buttons']:
                self.toolBar.addAction(self.slitsAction)
                self.toolBar.addSeparator()


        self.contrastAction = QtGui.QAction(
            QtGui.QIcon(os.path.join(self.iconsDir, 'contrast.png')),
            'Enhance Contrast', self)
        self.contrastAction.setShortcut('Alt+C')
        self.contrastAction.setIconText('Enhance Contrast')
        self.contrastAction.triggered.connect(self.toggle_contrast)
        self.contrastAction.setCheckable(True)
        if 'contrast' in self.appConfig['buttons']:
            if 'contrastEnhanced' in self.cameraConfig:
                self.contrastAction.setChecked(
                        self.cameraConfig['contrastEnhanced'])
            self.toolBar.addAction(self.contrastAction)
            self.toolBar.addSeparator()

        self.cmapAction = QtGui.QAction(
            QtGui.QIcon(os.path.join(self.iconsDir, 'cmap2.png')),
            'Apply Colormap', self)
        self.cmapAction.setShortcut('Alt+M')
        self.cmapAction.setIconText('Apply Colormap')
        self.cmapAction.triggered.connect(self.toggle_cmap)
        self.cmapAction.setCheckable(True)
        if 'colormap' in self.appConfig['buttons']:
            if 'cmapEnabled' in self.cameraConfig:
                self.cmapAction.setChecked(self.cameraConfig['cmapEnabled'])
            self.toolBar.addAction(self.cmapAction)
            self.toolBar.addSeparator()
        else:
            self.cmapAction.setChecked(False)

        self.histAction = QtGui.QAction(
            QtGui.QIcon(os.path.join(self.iconsDir, 'hist.png')),
            'Show Histogram', self)
        self.histAction.setShortcut('Alt+H')
        self.histAction.setIconText('Show Histogram')
        self.histAction.setCheckable(True)
        self.histAction.triggered.connect(self.toggle_hist)
        if 'hist' in self.appConfig['buttons']:
            if 'histEnabled' in self.cameraConfig:
                self.histAction.setChecked(self.cameraConfig['histEnabled'])
            self.toolBar.addAction(self.histAction)
            self.toolBar.addSeparator()
        else:
            self.histAction.setChecked(False)

        saveAction = QtGui.QAction(
            QtGui.QIcon(os.path.join(self.iconsDir, 'save.png')),
            'Save Image', self)
        saveAction.setShortcut('Ctrl+S')
        saveAction.setIconText('Save Image')
        saveAction.triggered.connect(self.save_image)

        self.toolBar.addAction(saveAction)

        setupAction = QtGui.QAction(
            QtGui.QIcon(os.path.join(self.iconsDir, 'setup.png')),
            'Setup Image', self)
#        setupAction.setShortcut('Ctrl+S')
        setupAction.setIconText('Setup Image')
        setupAction.triggered.connect(self.setup_image)

        self.toolBar.addAction(setupAction)

    def setup_image(self):
        openDialog = SetupDialog(copy.copy(self.cameraConfig),
                                 copy.copy(self.rawFrame))
        openDialog.applyConfig.connect(self.procObj.change_config)
        openDialog.applyConfig.connect(self.update_config)

        result = openDialog.exec_()

    def closeEvent(self, event):
        self.listenerThread.terminate()
        self.processThread.terminate()
        event.accept()

    def save_image(self):
        try:
            now = datetime.datetime.now()
            #  Set proper path and filename template
            filename = os.path.join(self.cameraConfig['savedir'],
                                    self.cameraConfig['name']+"_"+now.strftime("%Y-%m-%d_%H-%M-%S")+".png")
            cv2.imwrite(filename, self.currentImage)
            print("Saved to", filename)
        except:
            raise
            print("Couldn't save image")

    def draw_slits(self, img, width, height):
        slitL, slitR, slitT, slitB = self.slits
        gridL, gridR, gridT, gridB = self.gridLimits

        try:
            slitLpos = slitL.get()
            slitRpos = slitR.get()
            slitTpos = slitT.get()
            slitBpos = slitB.get()

            coordL = int((slitLpos + gridR)/float(-gridL+gridR)*width)
            coordR = int((slitRpos + gridR)/float(-gridL+gridR)*width)
            coordT = int((slitTpos + gridT)/float(-gridB+gridT)*height)
            coordB = int((slitBpos + gridT)/float(-gridB+gridT)*height)

            cv2.rectangle(img, (0, 0), (width-coordL, height), (0, 255, 255), -1)
            cv2.rectangle(img, (width-coordR, 0), (width, height), (0, 255, 255), -1)
            cv2.rectangle(img, (0, 0), (width, height-coordT), (0, 0, 255), -1)
            cv2.rectangle(img, (0, height-coordB), (width, height), (0, 0, 255), -1)

            return img
        except:
            return img

    def process_frame(self, data):
        self.rawFrame = data

    def update_config(self, config):
        isInitRequested = False
        if config['name'] != self.cameraConfig['name']:
            self.setWindowTitle(config['name'])
        if config['address'] != self.cameraConfig['address'] or\
            config['type'] != self.cameraConfig['type']:
                isInitRequested = True
        if isGigE and config['frame_rate'] != self.cameraConfig['frame_rate']:
            isInitRequested = True
        self.cameraConfig = config

        try:
            with open(self.configFile, "w") as write_file:
                json.dump(self.cameraConfig, write_file)
        except:
            print("No file name specified")

        self.update_geometry()
#        print()
        if isInitRequested:
            self.listener.listening = False
            self.listener.cameraConfig = self.cameraConfig
            self.reInit.emit()
#            self.listener.listening = True

    def update_geometry(self):
        self.frame_width, self.frame_height = self.cameraConfig['frame_size']
        self.hHistBase = np.linspace(self.gridLimits[0], self.gridLimits[1],
                                     self.frame_width)
        self.vHistBase = np.linspace(self.gridLimits[2], self.gridLimits[3],
                                     self.frame_height)
        self.vHistWidget.setFixedSize(self.histogramHeight, self.frame_height)
        self.hHistWidget.setFixedSize(self.frame_width, self.histogramHeight)
        self.gView.setFixedSize(self.frame_width, self.frame_height)
        self.vb.setRange(QtCore.QRectF(0, 0, self.frame_width, self.frame_height),
                    padding=0)
        self.setFixedSize(self.frame_width+40+self.toolBarHeight+self.histogramHeight,
                          self.frame_height+25+self.histogramHeight) #+self.toolBarHeight)

    def update_frame(self, data):
        self.currentImage = data
        self.img.setImage(cv2.cvtColor(data, cv2.COLOR_BGR2RGB),
                          autoLevels=False, autoDownsample=True) #False)

    def update_hist(self, histList):
        self.hHist.setData(y=histList[0], x=self.hHistBase)
#        self.vHist.setData(x=histList[1], y=self.vHistBase)
        self.vHist.setData(y=histList[1], x=self.vHistBase)

        self.hHistWidget.autoRange(padding=0)
        self.vHistWidget.autoRange(padding=0)
        self.vHistWidget.showGrid(x=False, y=True)
        self.hHistWidget.showGrid(x=True, y=False)
