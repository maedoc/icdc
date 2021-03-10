from .qt import QtCore, QtGui


class Status(QtGui.QStatusBar):
    def __init__(self, parent=None):
        QtGui.QStatusBar.__init__(self, parent=parent)
