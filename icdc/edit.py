import numpy as np
from scipy import signal

from pyqtgraph.parametertree import Parameter, ParameterTree

from sklearn.decomposition import FastICA

from .qt import QtCore, QtGui, pg, qt_core_enum
from .core import Action, ParamsDialog
from .util import ChannelSelect


def _filter_signal(b, a, data):
    for i, y in enumerate(data):
        data[i] = signal.filtfilt(b, a, y)
    return data


preprocess_parameters = [
    {"name": "Diff", "type": "bool", "value": False},
    {"name": "High-pass frequency", "type": "float", "value": 5.0},
    {"name": "Low-pass frequency", "type": "float", "value": 100.0},
]


class PreprocessParTree(ParameterTree):
    def __init__(self, parent=None):
        ParameterTree.__init__(self, parent=parent)
        self.pars = Parameter.create(
            name="detect", type="group", children=preprocess_parameters
        )
        self.setParameters(self.pars, showTop=False)


class PreprocessSettingsDialog(QtGui.QDialog):
    def __init__(self, parent=None):

        QtGui.QDialog.__init__(self, parent=parent)

        self.lay = QtGui.QVBoxLayout()
        self.setLayout(self.lay)

        self.partree = PreprocessParTree()
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
        return [p.value() for p in self.partree.pars.children()]


class Preprocess(Action):
    _text = "&Filter data"
    _shortcut = "F4"

    def run(self):
        pw = PreprocessSettingsDialog(parent=self.main)
        self.status.emit("getting settings from user...")
        ret = pw.exec_()
        if ret:
            diff, lo, hi = pw.values
            msg = "filtering diffing %r, high pass freq %r Hz, low pass freq %r Hz" % (
                diff,
                lo,
                hi,
            )
            self.status.emit(msg)
            ds = self.main.dataset
            ds.data = ds.data.copy()
            if diff:
                ds.data = np.diff(ds.data)
                ds.nsamp -= 1
            b, a = signal.butter(3, [2 * lo / ds.fs, 2 * hi / ds.fs], "bandpass")
            ds.data[ds.pick_mask] = _filter_signal(b, a, ds.data[ds.pick_mask])
            ds.preprocess = (
                "Diff" if diff else "No diff"
            ) + ", band pass between %.2f and %.2f Hz" % (lo, hi)
            self.status.emit("done!")
            self.main.sig_data_changed.emit(ds)
        else:
            self.status.emit("preprocessing canceled")


class ICA(Action):
    _text = "Apply &ICA"

    def run(self):
        cfg = ParamsDialog(
            parent=self.main,
            parameters=[
                {"name": "N. components (-1 for all)", "type": "int", "value": -1},
                {
                    "name": "Algorithm",
                    "type": "list",
                    "values": ["parallel", "deflation"],
                    "value": "parallel",
                },
                {
                    "name": "Neg-entropy approx function",
                    "type": "list",
                    "values": ["logcosh", "exp", "cube"],
                    "value": "logcosh",
                },
                {
                    "name": "Max iterations",
                    "type": "int",
                    "value": 200,
                    "limits": (1, 1000000),
                    "step": 20,
                },
                {"name": "Tolerance", "type": "float", "value": 1e-4},
            ],
        )
        if cfg.exec_():
            ds = self.main.dataset
            ncomp, algo, negent, maxiter, tol = cfg.values
            msg = "applying fastica with %r, be patient!" % (cfg.values,)
            self.status.emit(msg)
            if ncomp < 1:
                ncomp = None
            ds.ica = FastICA(
                n_components=ncomp,
                algorithm=algo,
                fun=negent,
                max_iter=maxiter,
                tol=tol,
            )
            ds.data = ds.ica.fit_transform(ds.data.T).T
            ds.nchan = ds.data.shape[0]
            ds.labels = ["ICA%03d" % (i,) for i in range(ds.nchan)]
            ds.picks = ds.labels


class Bipolarize(Action):
    _text = "&Bipolarize sEEG montage"


class MergeEvents(Action):
    _text = "&Merge events"


class RejectEvents(Action):
    _text = "&Reject events"


class PickChannels(Action):
    _text = "P&ick channels"

    def run(self):
        ds = self.main.dataset
        cs = ChannelSelect(ds.labels, parent=self.main)
        for p in ds.picks:
            cs.cbs[p].setCheckState(
                qt_core_enum("CheckState", "Checked")
            )  # QtCore.Qt.CheckState.Checked)
        ret = cs.exec_()
        if ret:
            ds.picks = cs.selected


class ClearEvents(Action):
    _text = "C&lear events"

    def run(self):
        self.main.dataset.events = []


class RemoveSelectedEvents(Action):
    _text = "Remove &selected events"

    def run(self):
        tab = self.main.summary.tab_events
        ds = self.main.dataset
        ur = np.unique([i.row() for i in tab.selectedIndexes()])
        toremove = []
        for r in ur:
            id = tab.item(r, 0).value
            toremove.append(ds.events[id])
        list(map(ds.events.remove, toremove))


class SpatialWhiten(Action):
    _text = "&Whiten spatially"

    def run(self):
        ds = self.main.dataset
        self.status.emit("applying spatial whitening")
        for i, y in enumerate(ds.data):
            q1, q2, q3 = np.percentile(y, [33.33, 50, 66.66])
            qtd = ((q3 - q2) + (q2 - q1)) / 2.0
            ds.data[i] = (y - q2) / qtd


class Decimate(Action):
    _text = "&Decimate"

    def run(self):
        cfg = ParamsDialog(
            parent=self.main,
            parameters=[
                {
                    "name": "Decimation factor",
                    "type": "int",
                    "value": 5,
                    "limits": (1, 100),
                }
            ],
        )
        if cfg.exec_():
            ds = self.main.dataset
            factor, = cfg.values
            ds.fs /= factor
            ds.data = ds.data[:, ::factor]
            ds.nsamp = ds.data.shape[1]


actions = [
    Preprocess,
    Decimate,
    SpatialWhiten,
    ICA,
    Bipolarize,
    PickChannels,
    "separator",
    MergeEvents,
    RemoveSelectedEvents,
    RejectEvents,
    ClearEvents,
]
