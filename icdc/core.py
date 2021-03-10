import sys
import copy
import traceback

import numpy as np

from .qt import pg, QtCore, QtGui

from pyqtgraph.parametertree import Parameter, ParameterTree


class Dataset(object):
    "Aggregate dataset: data, fs, chlabels, nchan, nsamp, events, events labels, parameters, etc. "

    filename = ""
    fs = 0.0
    nchan = 0
    nsamp = 0

    def __init__(self, filename=""):
        self.filename = filename
        self.events = []
        self.labels = []
        self.picks = []
        self.preprocess = "No preprocessing"
        self.data = np.array(())

    @property
    def nev_per_chan(self):
        nev = [0 for _ in self.labels]
        for ev in self.events:
            for targ in ev["targets"]:
                if type(targ) in (str, str):
                    try:
                        targ = self.labels.index(targ)
                    except:
                        continue
                nev[targ] += 1
        return nev

    def copy(self):
        "Return a copy of this dataset"
        ds = Dataset(self.filename)
        for attr in "fs nchan nsamp events labels".split():
            setattr(ds, attr, copy.deepcopy(getattr(self, attr)))
        ds.data = self.data.copy()
        return ds

    @property
    def pick_mask(self):
        return np.array([l in self.picks for l in self.labels])

    @property
    def pick_idx(self):
        return np.r_[: self.nchan][self.pick_mask]

    def update(self, ds):
        samefile = self.filename == ds.filename
        for attr in "filename fs nchan nsamp labels data".split():
            setattr(self, attr, getattr(ds, attr))
        [self.events.append(e) for e in ds.events]
        if not samefile:
            self.picks = self.labels[:]


class ErrorBox(QtGui.QWidget):
    def __init__(self, msg, namespace=None, parent=None):
        QtGui.QWidget.__init__(self, parent=parent)
        self.msg = msg
        self.namespace = namespace or {}

        self.lay = QtGui.QVBoxLayout()
        self.setLayout(self.lay)

        self.setWindowTitle("Oh noes! An error!")

        self.textedit = QtGui.QTextEdit()
        self.textedit.setHtml("<pre>%s</pre>" % (msg,))
        self.lay.addWidget(self.textedit)

        self.lay_buttons = QtGui.QHBoxLayout()
        self.lay.addLayout(self.lay_buttons)

        self.pb_debug = QtGui.QPushButton("&Debug")
        self.pb_debug.clicked.connect(self.debug)
        self.lay_buttons.addWidget(self.pb_debug)

        self.pb_copy = QtGui.QPushButton("&Copy message to clipboard")
        self.pb_copy.clicked.connect(self.copy_msg)
        self.lay_buttons.addWidget(self.pb_copy)

        self.pb_close = QtGui.QPushButton("&OK")
        self.pb_close.clicked.connect(self.close)
        self.lay_buttons.addWidget(self.pb_close)

    def copy_msg(self):
        qc = QtGui.QClipboard()
        qc.setText(self.msg)

    def debug(self):
        import pyqtgraph.console as con

        ns = self.namespace["tb"].tb_next.tb_frame.f_locals
        self.cw = con.ConsoleWidget(namespace=ns)
        self.cw.show()

    def sizeHint(self):
        return QtCore.QSize(400, 400)


class Action(QtGui.QAction):
    _text = "<no name Action>"

    status = QtCore.Signal(object)
    done = QtCore.Signal()

    def __init__(self, parent=None):
        QtGui.QAction.__init__(self, self._text, parent)
        self.triggered.connect(self._run)
        self.status.connect(self.main.status)
        self.done.connect(self.main.status_done)
        if hasattr(self, "_shortcut"):
            self.setShortcut(QtGui.QKeySequence(self._shortcut))

    def _run(self):
        try:
            self.status.emit("starting " + self._text)
            self.run()
            self.done.emit()
        except Exception as exc:
            t, v, tb = sys.exc_info()
            msg = "%r failed\n%s\n\n%s" % (
                self,
                "\n".join(traceback.format_tb(tb)),
                exc,
            )
            self.status.emit(msg)
            self.excmsg = ErrorBox(msg, namespace=locals())
            self.excmsg.show()
        finally:
            m = self.main
            m.sig_data_changed.emit(m.dataset)

    def run(self):
        "Subclasses should override this method to provide action code"
        msg = "An %s instance was triggered but no handler is defined."
        self.status.emit(msg % (self.__class__.__name__,))

    def _msgbox(self, msg, info=""):
        self.msg = QtGui.QMessageBox.information(
            self.parentWidget(), self._text, msg + info
        )

    @property
    def main(self):
        #           Menu           MenuBar        Main
        return self.parentWidget().parentWidget().parentWidget()


class ParamsDialog(QtGui.QDialog):
    parameters = []

    def __init__(self, parameters=[], parent=None):
        QtGui.QDialog.__init__(self, parent=parent)

        self.lay = QtGui.QVBoxLayout()
        self.setLayout(self.lay)

        self.pars = Parameter.create(
            name="pars", type="group", children=parameters or self.parameters
        )
        self.partree = ParameterTree()
        self.partree.setParameters(self.pars, showTop=False)

        self.lay.addWidget(self.partree)

        self.lay_buttons = QtGui.QHBoxLayout()
        self.lay.addLayout(self.lay_buttons)

        self.pb_no = QtGui.QPushButton("&Cancel")
        self.pb_no.clicked.connect(self.reject)
        self.lay_buttons.addWidget(self.pb_no)

        self.pb_ok = QtGui.QPushButton("&OK")
        self.pb_ok.clicked.connect(self.accept)
        self.lay_buttons.addWidget(self.pb_ok)
        self.pb_ok.setDefault(True)

    @property
    def values(self):
        return [p.value() for p in self.pars.children()]
