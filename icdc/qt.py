import sys

from pyqtgraph import QtCore, QtGui
import pyqtgraph as pg

if sys.platform == "win32":
    print("deactivating pyqtgraph's use of weave")
    pg.functions.USE_WEAVE = False
    pg.setConfigOptions(useWeave=False)


def qt_core_enum(obj, attr):
    try:
        # PySide style
        return getattr(getattr(QtCore.Qt, obj), attr)
    except:
        pass
    # PyQt style
    return getattr(QtCore.Qt, attr)
