"""
Interactive selectors and viewers?

"""

import time

import numpy as np
from scipy import stats

import pyqtgraph as pg
from pyqtgraph import QtCore as qc, QtGui as qg
from pyqtgraph import console

from . import stat

# from . import util


class CloudPicker(pg.PlotWidget):
    """
    Widget to display a points in 2D and a ROI to select points of interest.

    - A ScatterItem shows points in 2D
    - A mouse-draggable circle selects points
    - If selection changes, sigMaskChanged is emitted with the mask

    """

    sigMaskChanged = qc.Signal(object)

    def __init__(self, xy, labels=None, hlabel=None, parent=None):
        "Create CloudPicker from (n_obs, 2+) size array of points"

        pg.PlotWidget.__init__(self, parent=parent)

        # unpack two dimensions of input data to x & y
        self.x, self.y = self.xy = xy[:2]

        # class labels optional
        # hlabel will be highlighted, rest not
        if labels is not None and hlabel is not None:
            _ = labels.copy()
            _[labels == hlabel] = 0
            _[labels != hlabel] = 1
            labels = _
        if labels is None:
            labels = np.zeros(self.x.size, np.int32)
        self.labels = labels
        self.ulabels = np.unique(labels)

        self.setup_plot()
        self.setup_roi()

    def setup_plot(self):
        self.plot = self  # .plot()
        self.showGrid(x=True, y=True)

        # don't show the default left & bottom axes
        [self.plot.showAxis(o, False) for o in ("left", "bottom")]

        # create & add scatter item to plot
        self.scatters = []
        for i, ul in enumerate(self.ulabels):
            mask = self.labels == ul
            brush = pg.intColor(i, hues=len(self.ulabels), alpha=125)
            spi = pg.ScatterPlotItem(
                x=self.x[mask], y=self.y[mask], pen=None, brush=brush
            )
            self.scatters.append(spi)

        list(map(self.plot.addItem, self.scatters))

    def setup_roi(self):
        self.roi = pg.CircleROI([0, 0], [1, 1])
        self.plot.addItem(self.roi)
        self.roi.sigRegionChanged.connect(self.check_mask_change)

    _old_mask = None

    def check_mask_change(self, roi):
        "See if mask change, if so, emit sigMaskChanged"
        new = self.compute_mask()
        if self._old_mask is not None and (new == self._old_mask).all():
            return
        self._old_mask = new
        self.sigMaskChanged.emit(new)

    def compute_mask(self):
        "Compute & return bool mask selecting points inside roi"
        (cx, cy), (r, _) = self.roi.pos(), self.roi.size()
        return np.sqrt((self.x - cx) ** 2 + (self.y - cy) ** 2) <= r


class CloudStats(pg.PlotWidget):
    """
    Widget to display epoch stats (interactively)

    - Holds a 2/3D array of epoch data
    - Plot with 2D image item
    - On update, stats across epochs calculated
    - Interactive update is rate limited, does partial then full update

    If epochs are 2D, assume we're showing a density plot

    """

    alpha = 0.05
    n_ep_fast = 200

    def __init__(self, fs, ep, labels, parent=None):
        pg.PlotWidget.__init__(self, parent=None)
        self.fs = fs
        self.ep = ep
        self.labels = labels
        self.setup_lut()
        self.setup_plot()

    def setup_lut(self):
        "Create typical blue neg, black zero, red positive color map"
        self.lut = np.r_[np.zeros(128), np.r_[:128]] * 2
        self.lut = np.c_[self.lut, self.lut * 0.0, self.lut[::-1]].astype(np.ubyte)

    def setup_plot(self):
        "Scaffold plot and image item"
        self.plot = self  # .plot()
        self.showGrid(x=True, y=True)

        if self.ep.ndim == 3:
            self.plot.getAxis("left").setTicks(
                [[(i + 0.5, l) for i, l in enumerate(self.labels)]]
            )
            self.plot.getAxis("left").setWidth(100)

        wnsamp = self.ep.shape[1]
        winsz = wnsamp * 1.0 / self.fs
        xts = [
            (int(wnsamp * ph), "%0.3f" % ((ph - 0.5) * winsz,))
            for ph in [0.0, 0.5, 1.0]
        ]
        self.plot.getAxis("bottom").setTicks([xts])

        # self.plot.getAxis('left').setTicks(util.seeg_major_minor_ch_labels(self.labels))

        self.image = pg.ImageItem()
        if self.ep.ndim == 3:
            self.image.setLookupTable(self.lut)
        self.plot.addItem(self.image)

    def select_ep(self, mask, skip=True):
        "Select epochs on mask, maybe skipping some if too many"
        if skip:
            mask = np.r_[: mask.size][mask][:: (mask.sum() / self.n_ep_fast + 1)]
        return self.ep[mask]

    def compute_map(self, epi):
        "Compute stat map for display from mask"
        _, P = stats.ttest_1samp(epi, 0.0, axis=0)
        PT, _ = stat.fdr(P, alpha=self.alpha)
        return (P < PT) * epi.mean(axis=0)

    _level_scale = 1.0

    def update_image(self, mask, skip=True):
        "Update stat map on displayed image based on mask"
        epi = self.select_ep(mask, skip=skip)
        if epi.ndim == 3:
            std = epi.std() / self._level_scale
            self.image.setLevels([std, -std])
            self.image.setImage(self.compute_map(epi), autoLevels=False)
        elif epi.ndim == 2:
            T = np.tile(np.r_[: epi.shape[1]], (epi.shape[0], 1)).ravel()
            Y = epi.ravel()
            q5, q95 = np.percentile(Y, [0.1, 99.9])
            H, _, _ = np.histogram2d(T, np.clip(Y, q5, q95), (epi.shape[1], 200))
            self.image.setImage(np.log(H + 1))
        else:
            raise ValueError("ep must have 2 or 3 dim, found %d" % (epi.ndim,))

    # help with interactive update
    _full_timer = None
    _last_update = 0.0

    def interactive_update(self, mask):
        "Update map, first quickly, then with full stats"

        # rate limit interactive updates to 5 hz
        now = time.time()
        if now - self._last_update < 0.2:
            return
        self._last_update = now

        # cancel pending full update timer
        if self._full_timer is not None:
            self._full_timer.cancel()

        # partial update
        self.update_image(mask)

        # schedule full update after 200 ms
        self._full_timer = qc.QTimer.singleShot(
            200, lambda: self.update_image(mask, skip=False)
        )


class ClassAtlas(qg.QMainWindow):
    def __init__(self, fs, events, ep, xy, labels, ch_labels, parent=None):

        qg.QMainWindow.__init__(self, parent=parent)

        self.fs = fs

        self.split_main = qg.QSplitter(qc.Qt.Horizontal)
        self.setCentralWidget(self.split_main)
        self.setWindowTitle("Class Atlas")

        self.split_ctrl = qg.QSplitter(qc.Qt.Vertical)
        """
        self.cli = console.ConsoleWidget(namespace={'ca':self}, editor='gvim {fileName} +{lineNum}')
        self.split_ctrl.addWidget(self.cli)
        """
        self.b_add_class = qg.QPushButton("Add class")
        self.b_add_class.clicked.connect(self._b_add_class_cb)
        self.lay_ctrl = qg.QVBoxLayout()
        self.split_ctrl.setLayout(self.lay_ctrl)
        self.lay_ctrl.addWidget(self.b_add_class)
        self.lay_ctrl.insertStretch(-1)
        self.split_main.addWidget(self.split_ctrl)

        self.xy = xy
        self.events = events
        self.ep = ep
        self.labels = labels
        self.ch_labels = ch_labels

        self.split_views = qg.QSplitter(qc.Qt.Vertical)
        self.split_main.addWidget(self.split_views)

        self.lay_pickers = qg.QHBoxLayout()
        self.lay_stats = qg.QHBoxLayout()
        self.pw_timeline = pg.PlotWidget()

        self.w_lay_pickers = qg.QWidget()
        self.w_lay_pickers.setLayout(self.lay_pickers)
        self.w_lay_stats = qg.QWidget()
        self.w_lay_stats.setLayout(self.lay_stats)

        self.split_views.addWidget(self.w_lay_pickers)
        self.split_views.addWidget(self.w_lay_stats)
        self.split_views.setSizes([150, 700])
        # self.split_views.addWidget(self.pw_timeline)

        self.classes = []
        for i in np.unique(labels):
            self.add_class(hlabel=i)
            self.classes[-1]["stats"].interactive_update(labels == i)

    def _b_add_class_cb(self, *args):
        self.add_class()

    def add_class(self, hlabel=None):

        class_id = len(self.classes)

        # setup picker and dock
        picker = CloudPicker(self.xy, self.labels, hlabel=hlabel)
        self.lay_pickers.addWidget(picker)

        # setup stats and dock
        stats = CloudStats(self.fs, self.ep, self.ch_labels)
        self.lay_stats.addWidget(stats)

        # connect pick updates to stat display
        picker.sigMaskChanged.connect(stats.interactive_update)

        # store all this
        self.classes.append(locals())


class ScatterStat(pg.QtGui.QWidget):
    def __init__(self, *args, **kwds):
        pg.QtGui.QWidget.__init__(self)
        self.lay = pg.QtGui.QVBoxLayout()
        self.setLayout(self.lay)
        self.gw = pg.GraphicsWindow()  # .__init__(self, *args, **kwds)
        self.lay.addWidget(self.gw)
        self.rows = []
        self.lut = np.r_[np.zeros(128), np.r_[:128]] * 2
        self.lut = np.c_[self.lut, self.lut * 0.0, self.lut[::-1]].astype(np.ubyte)
        self.b_add_row = pg.QtGui.QPushButton("Add row with same data")
        self.b_add_row.clicked.connect(self._add_row_same_data)
        self.lay.addWidget(self.b_add_row)
        self.add_row(*args)
        self.n_ep_fast = kwds.pop("n_ep_fast", 50)
        self.alpha = kwds.pop("alpha", 0.05)
        self.show()

    def _add_row_same_data(self):
        self.add_row(*[self.rows[-1][k] for k in "xi labels ep ch_labels".split()])

    def add_row(self, xi, labels, ep, ch_labels):

        row_idx = len(self.rows)

        # prep data
        x, y = xi
        ulabels = np.unique(labels)
        emin, emax = ep.min(), ep.max()

        # plot embedded data, classes in different colors
        p_xi = self.gw.addPlot()
        [p_xi.showAxis(o, False) for o in ("left", "bottom")]
        s_xi = pg.ScatterPlotItem()
        for i, ulab in enumerate(ulabels):
            mask = labels == ulab
            s_xi.addPoints(
                x[mask], y[mask], pen=None, symbol="+"
            )  # , brush=(i, len(ulabels)))
        p_xi.addItem(s_xi)

        # setup roi
        r_xi = pg.CircleROI([0, 0], [1, 1])
        p_xi.addItem(r_xi)

        # setup stat plot
        p_stat = self.gw.addPlot()
        p_stat.getAxis("left").setTicks(
            [[], [(i + 0.5, l) for i, l in enumerate(ch_labels)]]
        )
        i_stat = pg.ImageItem()
        i_stat.setLookupTable(self.lut)
        p_stat.addItem(i_stat)

        # setup callback to update i_stat on r_xi selection of s_xi points
        tic = [time.time(), None]

        def update_stats_of_roi(roi, noskip=False):
            if time.time() - tic[0] < 0.1:
                return
            else:
                tic[0] = time.time()
            cx, cy = roi.pos()
            r, _ = roi.size()
            mask = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) <= r
            skip = 1 if noskip else mask.sum() / self.n_ep_fast + 1
            maski = np.r_[: mask.shape[0]][mask]
            epi = ep[mask[::skip]]
            std = epi.std()
            i_stat.setLevels([std, -std])
            try:
                _, P = stats.ttest_1samp(epi, 0.0, axis=0)
                PT, _ = stat.fdr(P, alpha=self.alpha)
                i_stat.setImage((P < PT) * epi.mean(axis=0), autoLevels=False)
            except Exception as e:
                print(
                    "%r row[%d] update_stats_of_roi failed with %r" % (self, row_idx, e)
                )

            # if we skipped data in fast interactive, wait 500 ms and update with full stats
            if skip > 1:

                def fullupdate():
                    update_stats_of_roi(r_xi, noskip=True)

                if tic[1] is not None:  # TODO REFACTOR
                    tic[1].stop()
                tic[1] = pg.QtCore.QTimer.singleShot(200, fullupdate)

        r_xi.sigRegionChanged.connect(update_stats_of_roi)

        self.rows.append(locals())
        self.gw.nextRow()
        return locals()
