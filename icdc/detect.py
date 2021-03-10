import functools

import numpy as np
from scipy import stats, signal, optimize

from pyqtgraph.parametertree import Parameter, ParameterTree

from .qt import QtCore, QtGui, pg
from .core import Action
from . import stat, util


def lfdr_detect(
    fs, y, q=1e-3, minel=0.0, maxel=0.5, rf=0.5, ht=False, dc=1, info=False
):
    """
    Perform detection on continuous signal `y`.

    This function filters the signal, estimates the LFDR (see `lfdr`), and
    identifies where `y` remains "unlikely" (LFDR < `q`), for a period of
    time longer than `minel` and shorter than `maxel`, with a refractory
    period of `rf`.
    
    `y` may be an array of signals, i.e. shape == (nchan, ntime) in which 
    case detection is performed on each channel, and a list is returned.


    Parameters
    ----------

    fs : float
        Sampling frequency of `y`
    y : array
        Continuously sampled signal on which to perform detection
    q : float
        Threshold on LFDR
    minel : float
        Minimum event length
    maxel : float
        Maximum event length
    rf : float
        Refractory period of events
    info : bool
        Whether to return all information in dictionary or just 
            event times

    Returns
    -------

    ret : dict or array
        An array of event peak times, or if `info` is `True`, a
            dictionary of the local variables.

    """

    fs = float(fs)

    t = np.r_[: len(y)] / fs

    # analyze distribution
    xb, f, cf, fdr, llx = stat.lfdr(y, dc=dc, doplot=False)

    # fdr tarnsform hilbert
    if ht:
        hy = abs(signal.hilbert(y))
        llh = np.interp(hy, xb, np.log(fdr))
    else:
        hy, llh = y, llx

    # generate events
    ev = np.c_[np.isfinite(llh), llh < np.log(q)].all(axis=1)
    ev[~np.isfinite(llh)] = True
    e0, = np.argwhere(np.c_[~ev[:-1], ev[1:]].all(axis=1)).T
    e1, = np.argwhere(np.c_[ev[:-1], ~ev[1:]].all(axis=1)).T

    if len(e0) == 0 or len(e1) == 0:
        return locals() if info else []

    # boundaries
    if e1[0] < e0[0]:
        e0 = np.r_[0, e0]
    if e0[-1] > e1[-1]:
        e1 = np.r_[e1, llh.shape[0] - 1]

    # compute event length and long-enough mask
    el = np.diff(np.c_[e0, e1])[:, 0] / fs
    le = np.c_[el > minel, el < maxel].all(axis=1)

    # pull out remaining peaks & align them
    peaks = []
    for i, (e0i, e1i) in enumerate(zip(e0[le], e1[le])):
        hi = np.abs(hy[e0i:e1i])
        peaks.append((hi.max(), t[e0i] + np.argmax(hi) / fs))

    if len(peaks) == 0:
        return locals() if info else []

    # mask refractory period
    ph, pt = np.array(peaks).T
    nonmask = []
    for i, (phi, pti) in enumerate(peaks):
        mask = np.c_[pt > pti - rf, pt < pti + rf].all(axis=1)
        if (phi >= ph[mask]).all():
            nonmask.append(i)

    nfpeak = np.array(peaks)[np.array(nonmask), 1]

    return locals() if info else nfpeak


def _lfdr_detect_single(arg):
    fs, yi, kwds = arg
    try:
        out = lfdr_detect(fs, yi, **kwds)
    except Exception as out:
        pass
    return out


def batch_lfdr_detect(fs, y, n_jobs=1, **kwds):
    """
    Perform lfdr detection on each row of `y` using `n_jobs` processes. `kwds`
    will be provided to the detection function. 

    """

    jobs = [(fs, yi, kwds) for yi in y]
    with util.mpool(n_jobs) as p:
        results = p.map(_lfdr_detect_single, jobs)
    return results


detection_parameters = [
    {"name": "Q value", "type": "float", "value": 1e-5},
    {
        "name": "Minimum event length",
        "type": "float",
        "value": 0.0,
        "siPrefix": True,
        "suffix": "s",
    },
    {
        "name": "Maximum event length",
        "type": "float",
        "value": 0.5,
        "siPrefix": True,
        "suffix": "s",
    },
    {
        "name": "Refractory period",
        "type": "float",
        "value": 0.2,
        "siPrefix": True,
        "suffix": "s",
    },
    {"name": "Decimate", "type": "int", "value": 5},
    {"name": "Use Hilbert transform", "type": "bool", "value": False},
]


class DetectParTree(ParameterTree):
    def __init__(self, parent=None):
        ParameterTree.__init__(self, parent=parent)
        self.pars = Parameter.create(
            name="detect", type="group", children=detection_parameters
        )
        self.setParameters(self.pars, showTop=False)


class DetectSettingsDialog(QtGui.QDialog):
    def __init__(self, parent=None):

        QtGui.QDialog.__init__(self, parent=parent)

        self.lay = QtGui.QVBoxLayout()
        self.setLayout(self.lay)

        self.partree = DetectParTree()
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


class SelectChannels(Action):
    _text = "&Select channels for detection"


class InteractiveDetection(Action):
    _text = "Open the &interactive detection tool"

    """
    Show detection parameters on left, and a few seconds of signal on the right, 
    changing parameters updates a informative visu.

    """


class RunDetect(Action):
    _text = "&Run standard detection"
    _shortcut = "F5"

    def run(self):
        dsd = DetectSettingsDialog(parent=self.main)
        self.status.emit("getting settings from user...")
        ret = dsd.exec_()
        if ret:
            ds = self.main.dataset
            q, minel, maxel, rf, dc, ht = dsd.values
            ds.detection = "detect with q=%g minel=%f maxel=%f rf=%f dc=%d ht=%s" % (
                q,
                minel,
                maxel,
                rf,
                dc,
                ht,
            )
            self.status.emit(ds.detection)
            ds = self.main.dataset
            events = batch_lfdr_detect(
                ds.fs,
                ds.data[ds.pick_mask],
                q=q,
                minel=minel,
                maxel=maxel,
                rf=rf,
                ht=ht,
                dc=dc,
                info=False,
            )
            pidx = ds.pick_idx
            for i, chev in enumerate(events):
                for ev in chev:
                    ds.events.append(
                        {
                            "label": "spike",
                            "value": 4200,
                            "time": ev,
                            "targets": [pidx[i]],
                        }
                    )


# TODO actions shouldn't be enabled unless possible..
#  how to manage chain of dependencies in workflow, and update..

actions = [RunDetect, SelectChannels, InteractiveDetection]
