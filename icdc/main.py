import sys

import numpy as np
from scipy import stats

from .qt import pg, QtCore, QtGui, qt_core_enum
from .menus import setup_menubar
from .status import Status
from .core import Dataset
from .view import WaveView


class TableWidget(pg.TableWidget):

    sort_col = 0
    sort_order = qt_core_enum(
        "SortOrder", "AscendingOrder"
    )  # QtCore.Qt.SortOrder.AscendingOrder

    def sortByColumn(self, col, order):
        pg.TableWidget.sortByColumn(self, col, order)
        self.sort_col = col
        self.sort_order = order

    def update_summary(self, ds):
        ary = self.summarize(ds)
        c, o = self.sort_col, self.sort_order
        self.setData(ary)
        self.sortByColumn(c, o)


class ChannelTable(TableWidget):
    def __init__(self, *args, **kwds):
        TableWidget.__init__(self, *args, **kwds)
        self.contextMenu.addAction("Add channel(s) to picks").triggered.connect(
            self.add_picks
        )
        self.contextMenu.addAction("Remove channel(s) from picks").triggered.connect(
            self.rm_picks
        )
        self.contextMenu.addAction(
            "Use selected channel(s) as picks"
        ).triggered.connect(self.use_picks)
        self.contextMenu.addAction("View channels' DWT").triggered.connect(
            self.pop_up_dwt
        )

    @property
    def selected_labels(self):
        ur = np.unique([i.row() for i in self.selectedIndexes()])
        return [self.item(r, 0).value for r in ur]

    def add_picks(self):
        for l in self.selected_labels:
            if l not in self.ds.picks:
                self.ds.picks.append(l)
        self.update_summary(self.ds)  # HACK

    def rm_picks(self):
        for l in self.selected_labels:
            if l in self.ds.picks:
                self.ds.picks.remove(l)
        self.update_summary(self.ds)  # HACK

    def use_picks(self):
        self.ds.picks = self.selected_labels
        self.update_summary(self.ds)  # HACK

    def sizeHint(self):
        return QtCore.QSize(100, 1000)

    _dwts = []

    def pop_up_dwt(self):
        ds = self.ds
        sl = self.selected_labels
        for l in sl:
            i = ds.labels.index(l)
            self._dwts.append(WaveView(ds.fs, ds.data[i]))

    def summarize(self, ds):
        self.ds = ds
        return np.array(
            list(
                zip(
                    ds.labels,
                    stats.kurtosis(
                        ds.data[:, :: (ds.data.shape[1] // 1000 + 1)], axis=1
                    ),
                    ds.nev_per_chan,
                    ds.pick_mask,
                )
            ),
            dtype=[
                ("Channel", object),
                ("Kurtosis", float),
                ("N. events", int),
                ("Picked", bool),
            ],
        )


class EventTable(TableWidget):
    def summarize(self, ds):
        es = []
        for id, e in enumerate(ds.events):
            val = e.get("value", -1)
            targets = e.get("targets", [])
            if targets:
                if type(targets[0]) not in (str, str):
                    targets = [ds.labels[i] for i in targets]
            es.append((id, e["label"], e["time"], val, ", ".join(targets)))
        print("event table found %d events, %r" % (len(es), repr(ds.events)[:40]))
        return np.array(
            es,
            dtype=[
                ("Id", int),
                ("Label", object),
                ("Time", float),
                ("Value", int),
                ("Target channels", object),
            ],
        )


class DatasetSummary(QtGui.QWidget):
    "Widget summarizing data"

    """
    Instead of a compelte generate display, it could be useful to show some
    summary statistics, spectrum, kurtosis, as well as table of channels, with
    detections.

    """

    status = QtCore.Signal(object)

    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent=parent)

        self.lay = QtGui.QVBoxLayout()
        self.setLayout(self.lay)

        for k in "filename basic preprocess detect classify event".split():
            key = "lb_" + k
            setattr(self, key, QtGui.QLabel(""))
            self.lay.addWidget(getattr(self, key))

        self.lay.addStretch()

        self.tabs = QtGui.QTabWidget()
        self.lay.addWidget(self.tabs)

        self.tab_chans = ChannelTable()
        self.tabs.addTab(self.tab_chans, "Channels")

        self.tab_events = EventTable()
        self.tabs.addTab(self.tab_events, "Events")

    def update_summary(self, dataset):
        ds = dataset
        if ds.filename:
            self.status.emit("updating datset summary...")
            self.lb_filename.setText(ds.filename)
            self.lb_basic.setText(
                "%d chans x %d samples @ %.3f Hz" % (ds.nchan, ds.nsamp, ds.fs)
            )
            ncl = len(np.unique(getattr(ds, "evlabels", [])))
            self.lb_event.setText("%d events, %d classes" % (len(ds.events), ncl))
            self.lb_preprocess.setText(str(ds.preprocess))
            self.lb_detect.setText(str(getattr(ds, "detection", "")))
            self.lb_classify.setText(str(getattr(ds, "classification", "")))
            self.tab_chans.update_summary(dataset)
            self.tab_events.update_summary(dataset)
            self.status.emit("ready")
        else:
            self.lb_filename.setText("Please open a file.")


class StdRedirect(object):
    def __init__(self, target):
        self.target = target

    def write(self, msg, *args):
        msg %= args
        self.target(msg)


class Main(QtGui.QMainWindow):
    "Main window of ICDC app"

    main_windows = []
    sig_data_changed = QtCore.Signal(object)

    def __init__(self, dataset=None, parent=None):
        QtGui.QMainWindow.__init__(self, parent=parent)
        self.setWindowTitle("IC/DC")
        self.sb = Status(parent=self)
        self.setStatusBar(self.sb)
        setup_menubar(self.menuBar(), parent=self)
        self.sb.showMessage("Ready")
        self.main_windows.append(self)
        self.dataset = dataset or Dataset()
        self.summary = DatasetSummary()
        self.setCentralWidget(self.summary)
        self.sig_data_changed.connect(self.summary.update_summary)
        self.sig_data_changed.emit(self.dataset)
        self.log = []
        # sys.stdout = sys.stderr = StdRedirect(self.status)

    def sizeHint(self):
        return QtCore.QSize(400, 600)

    def clone_window(self):
        Main(dataset=self.dataset.copy()).show()

    def status(self, msg):
        print(msg)
        self.current_status = msg
        self.sb.showMessage(msg)
        self.log.append(msg)

    def status_done(self):
        self.status(self.current_status + " ... done!")
