"""
Visualizers and corresponding actions

"""

import time
import numpy as np
import pyqtgraph as pg
import pywt
from .qt import QtCore, QtGui, qt_core_enum
from .core import Action

# from .wavedec import WaveView
from .d3 import MultiXImage
from . import atlas


class PagingViewBox(pg.ViewBox):
    def mouseMoveEvent(self, ev):
        print(ev)

    def mouseReleaseEvent(self, ev):
        print(self.viewRange())
        pg.ViewBox.mouseReleaseEvent(self, ev)


class MinutesSecondsAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        out = []
        s = 1
        m = 60 * s
        for value in values:  # map(int, values):
            nm = value / m
            ns = value % m
            out.append("%dm%.3f" % (nm, ns) if nm > 0 else "%.3fs" % (ns,))
        return out


class TimeSeriesWidget(pg.PlotWidget):
    def __init__(self, fs, data, **kwds):
        labels = kwds.pop("labels", [])
        self.yax = pg.AxisItem(orientation="left")
        self.xax = MinutesSecondsAxis(orientation="bottom")
        self.yax.setTicks([[], [(i, l) for i, l in enumerate(labels)]])
        kwds["axisItems"] = {"left": self.yax, "bottom": self.xax}
        show = kwds.pop("show", True)

        self.pvb = PagingViewBox()
        kwds["viewBox"] = self.pvb

        pg.PlotWidget.__init__(self, **kwds)

        self.setDownsampling(auto=True, mode="peak")
        self.setClipToView(True)
        self.setXRange(0, 10)
        self._fs = fs
        self._data = data
        self._dstd = data.std()
        self.t = np.r_[: data.shape[1]] * 1.0 / fs
        self.scale = 0.5
        self.lines = []
        for _ in range(len(data)):
            pdi = pg.PlotDataItem()
            self.addItem(pdi)
            self.lines.append(pdi)
        self.update_lines()
        self.showGrid(x=True, y=True)
        if show:
            self.show()

    def update_data(self, data):
        self._data = data
        self.update_lines()

    def update_lines(self):
        for i, (line, y) in enumerate(zip(self.lines, self._data)):
            line.setData(self.t, y / self._dstd * self.scale + i, antialias=True)

    def wheelEvent(self, ev):
        self.scale *= 1 + 0.2 * (1 if ev.delta() > 0 else -1)
        self.update_lines()

    dd = 0.8

    def keyReleaseEvent(self, ev):

        koi = "Left Right Up Down Minus Plus Home End"
        l, r, u, d, m, p, h, e = [
            getattr(pg.QtCore.Qt, "Key_" + s) for s in koi.split(" ")
        ]
        ek = ev.key()

        dx, dy = 0, 0
        vb = self.getPlotItem().getViewBox()
        (xmin, xmax), (ymin, ymax) = vb.viewRange()

        # just arrow keys
        if ek in (l, r):
            dx = -1 if ek == l else 1
        elif ek in (u, d):
            if ev.modifiers() and pg.QtCore.Qt.Key_Control:
                self.scale *= 1 + 0.25 * (1 if ek == u else -1)
                return
            dy = -1 if ek == d else 1
        elif ek in (h, e):
            dx = -xmin if ek == h else (self.t[-1] - xmax)

        # shift + arrow
        if ev.modifiers() and pg.QtCore.Qt.Key_Shift:
            if ek in (l, r):
                dx = dx * (xmax - xmin) * self.dd
            elif ek in (u, d):
                dy = dy * (ymax - ymin) * self.dd

        vb.translateBy((dx, dy))


def compare(fs, labels, *data, **kwds):
    win = pg.QtGui.QSplitter()
    win.setOrientation(pg.QtCore.Qt.Vertical)
    kwds["show"] = False
    first_ts = None
    for i, datum in enumerate(data):
        ts = TimeSeriesWidget(fs, datum, labels=labels, **kwds)
        win.addWidget(ts)
        if i == 0:
            first_ts = ts
        else:
            ts.getPlotItem().setXLink(first_ts.getPlotItem())
    win.show()
    return win


def wtrect(n, h):
    rect = pg.QtCore.QRect()
    rect.setTop(h)
    rect.setBottom(h)
    rect.setLeft(0)
    rect.setRight(n)
    return rect


class Signal(pg.PlotItem):
    def __init__(self, fs, data, parent=None):
        pg.PlotItem.__init__(self, parent)
        self.time = np.r_[: data.size] * 1.0 / fs
        # HACK
        self.time = np.r_[: int(data.size * 1.0 / fs + 1) : 1j * data.size] * 1.0
        self.data = data
        self.pdi = pg.PlotDataItem(x=self.time, y=self.data)
        self.addItem(self.pdi)
        self.getAxis("left").setWidth(80)


class WaveDec(pg.PlotItem):
    def __init__(self, fs, data, wtype="db1", minfreq=2.0, maxfreq=150.0, parent=None):
        pg.PlotItem.__init__(self, parent)
        self.endtime = (len(data) - 1) * 1.0 / fs
        self.endtime = int(data.size * 1.0 / fs)
        self.data = data
        self.wtype = wtype
        self.compute_wavedec(wtype)
        self.ticks = []
        # self.wfs = [fs/(2**(len(self.cD)-i)) for i in len(self.cD)]
        for i, cdi in enumerate(self.cD):
            fi = fs / (2 ** (len(self.cD) - i))
            im = pg.ImageItem(
                np.abs(cdi[:, np.newaxis]) * cdi.size
            )  # /(2**(len(self.cD)-i)))
            im.setRect(wtrect(self.endtime, len(self.ticks)))
            self.addItem(im)
            self.ticks.append(
                (i + 0.5, "%.0f Hz" % (fi,) if fi > 1 else "%.0f s" % (1 / fi,))
            )
        self.getAxis("left").setTicks([self.ticks])
        self.getAxis("left").setWidth(80)

    def compute_wavedec(self, wtype):
        self.wavelet = pywt.Wavelet(wtype)
        tic = time.time()
        cs = pywt.wavedec(self.data, self.wavelet)
        print("wavedec required %.2f s" % (time.time() - tic,))
        self.cA, self.cD = cs[0], cs[1:]


class WaveView(pg.GraphicsWindow):
    def __init__(self, fs, signal, wtype="db1", minfreq=2.0, parent=None):
        pg.GraphicsWindow.__init__(self, parent)
        self.fs = fs
        self.signal = signal
        self.p_ts = Signal(fs, signal)
        self.p_ts.showGrid(x=1, y=1)
        self.addItem(self.p_ts)
        self.nextRow()
        self.p_wd = WaveDec(fs, signal, wtype=wtype, minfreq=minfreq)
        self.p_wd.setXLink(self.p_ts)
        self.addItem(self.p_wd)
        self.show()


class ShowSignals(Action):
    _text = "&Signals"

    def run(self):
        ds = self.main.dataset
        self.sigwin = compare(
            ds.fs, np.array(ds.labels)[ds.pick_mask], ds.data[ds.pick_mask]
        )


class ShowDetection(ShowSignals):
    _text = "&Detections"

    def run(self):
        ShowSignals.run(self)
        ds = self.main.dataset
        self.tsw = self.sigwin.children()[0]
        self.tsw.evitem = pg.ScatterPlotItem()
        self.pidx = {ip: i for i, ip in enumerate(ds.pick_idx)}
        pos = []
        for e in ds.events:
            targ = e["targets"]
            if targ:
                targ = targ[0]
                if type(targ) in (str, str):
                    if not any(targ == l for l in ds.labels):
                        continue
                    targ = ds.labels.index(targ)
                pos.append((e["time"], self.pidx[targ]))
        self.tsw.evitem.setData(
            pos=np.array(pos),
            # TODO parameters
            symbol="+",
            brush=None,
            pen="r",
            size=5,
        )
        self.tsw.addItem(self.tsw.evitem)

        # TODO double click on event table highlights in view
        # self.main.summary.tab_events.double_click.connect(self.highlight_event)

    def highlight_event(self, ev):
        e = self.main.dataset.events[i]
        t, i = e["time"], self.pidx[e["targets"][0]]
        vb = self.tsw.getPlotItem().getViewBox()
        vb.setXRange(t - 0.5, t + 0.5, update=False)
        vb.setYRange(i - 2, i + 2, update=True)


class SignalCrossCorr(Action):
    _text = "Signal &cross correlation"

    def run(self):
        ds = self.main.dataset
        xc = np.corrcoef(ds.data[ds.pick_mask])
        self.im = pg.image(xc, title="Cross correlation")


class ImWin(QtGui.QMainWindow):
    def sizeHint(self):
        return QtCore.QSize(600, 600)


class EventSimilarity(Action):
    _text = "Event &similarity"

    def run(self):
        ds = self.main.dataset
        mw = ImWin(parent=self.main)
        im = pg.ImageView()
        mw.setCentralWidget(im)
        mw.setWindowTitle("Event similarity")
        im.setImage(ds.ss)
        mw.show()


class ClassRates(Action):
    _text = "Class &rates"

    def run(self):
        ds = self.main.dataset
        ts = np.array([e["time"] for e, o in zip(ds.events, ds.oob) if not o])
        win = pg.GraphicsWindow(title="Event class rates in time")
        self.win = win
        ul = np.unique(ds.evlabels)
        t0 = 0.0
        tf = ds.nsamp * 1.0 / ds.fs
        for i, uli in enumerate(ul):
            mask = ds.evlabels == uli
            counts, bins = np.histogram(ts[mask], bins=30, range=(t0, tf))
            p = win.addPlot(x=bins[:-1], y=counts, symbol="o")
            p.setLabel(axis="left", text="Class %s" % (i,))
            win.nextRow()
            if i == 0:
                self.firstplot = p
            else:
                p.setXLink(self.firstplot)


class ClassAtlas(Action):
    _text = "Class &atlas"

    def run(self):
        ds = self.main.dataset
        self.status.emit("starting class atlas..")
        events = np.array([e["time"] for o, e in zip(ds.oob, ds.events) if not o])
        ca = atlas.ClassAtlas(
            ds.fs,
            events,
            ds.epochs,
            ds.xi,
            ds.evlabels,
            np.array(ds.labels)[ds.pick_mask],
            parent=self.main,
        )
        ca.show()


class OpenDicom(Action):
    _text = "Open DIC&OM directory"

    def run(self):
        mxi = MultiXImage.from_file_dialog()
        if mxi:
            self.status.emit("starting DICOM viewer")
            mw = QtGui.QMainWindow(parent=self.main)
            mw.setCentralWidget(mxi)
            mw.setWindowTitle("DICOM Viewer")
            mw.show()
        else:
            self.status.emit("no directory selected!")


actions = [
    ShowSignals,
    SignalCrossCorr,
    "separator",
    ShowDetection,
    EventSimilarity,
    ClassAtlas,
    ClassRates,
    "separator",
    OpenDicom,
]
