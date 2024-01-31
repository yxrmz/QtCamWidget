#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 15:51:33 2020

@author: Roman Chernikov
"""

from pyqtgraph.Qt import QtGui
from QtCamWidget import VideoCamStreamer


if __name__ == "__main__":

    appConfig = {'buttons': ['grid', 'slits', 'contrast', 'colormap', 'hist'],
                 'iconSize': (48, 48)}

    qapp = QtGui.QApplication([])
    app = VideoCamStreamer(configFile='MainAxisCam.json', appConfig=appConfig)
    app.show()
    qapp.exec_()
