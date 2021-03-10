import numpy as np

from .qt import QtCore, QtGui
from .core import Action, ParamsDialog

from . import util, cluster


# "lower" level actions


class ParAction(Action):
    def __init__(self, *args, **kwds):
        super(ParAction, self).__init__(*args, **kwds)
        if self.__class__.run == ParAction.run:
            self.setDisabled(True)

    def run(self, parvals=None):
        if parvals:
            return parvals
        else:
            cfg = ParamsDialog(self._pars, parent=self.main)
            if cfg.exec_():
                return cfg.values


class ExtractEvents(ParAction):
    _text = "E&xtract events"
    _pars = [
        {"name": "All channels", "type": "bool", "value": True},
        {
            "name": "Window size",
            "type": "float",
            "value": 0.4,
            "limits": (0.0, 60.0),
            "step": 0.05,
            "suffix": "s",
            "siPrefix": True,
        },
    ]


class EventSimilarity(ParAction):
    _text = "Compute event &similarity"


class Cluster(ParAction):
    _text = "Compute event c&lusters"

    def run(self):
        cfg = ParamsDialog(
            parent=self.main,
            parameters=[
                {
                    "name": "Similarity dimension",
                    "type": "int",
                    "value": 2,
                    "limits": (1, 3),
                },
                {
                    "name": "Neighborhood quantile",
                    "type": "float",
                    "value": 0.05,
                    "limits": (0.0, 1.0),
                    "step": 0.01,
                },
                {
                    "name": "Min. n. class elements",
                    "type": "int",
                    "value": 20,
                    "limits": (2, 1000000),
                    "step": 10,
                },
            ],
        )
        if cfg.exec_():
            ds = self.main.dataset
            sdim, eps, minel = cfg.values
            self.status.emit("applying clustering %r" % (cfg.values,))
            if not hasattr(ds, "ss"):
                raise Exception("need to calculate similarity matrix first!")
            ds.xi, ds.evlabels = cluster.spectral_dbscan(
                ds.ss, n_dim=sdim, eps=eps, min_samples=minel
            )
            for e, o, el in zip(ds.events, ds.oob, ds.evlabels):
                e["value"] = 4199 if o else 4201 + el


# high level actions


class ClassTime(ParAction):
    _text = "&Temporal classification"

    def run(self):
        cfg = ParamsDialog(
            parent=self.main,
            parameters=[
                {
                    "name": "Window size (pre + post)",
                    "type": "float",
                    "value": 0.4,
                    "siPrefix": True,
                    "suffix": "s",
                    "step": 0.05,
                },
                {
                    "name": "Similarity measure",
                    "type": "list",
                    "values": ["corrcoef", "cov"],
                    "value": "corrcoef",
                },
                {
                    "name": "Use absolute value of measure",
                    "type": "bool",
                    "value": False,
                },
                {
                    "name": "Similarity dimension",
                    "type": "int",
                    "value": 2,
                    "limits": (1, 3),
                },
                {
                    "name": "Neighborhood quantile",
                    "type": "float",
                    "value": 0.05,
                    "limits": (0.0, 1.0),
                    "step": 0.01,
                },
                {
                    "name": "Min. n. class elements",
                    "type": "int",
                    "value": 20,
                    "limits": (2, 1000000),
                    "step": 10,
                },
            ],
        )
        if cfg.exec_():
            ds = self.main.dataset
            if len(ds.events) == 0:
                raise Exception("Did you run detection?!?")
            winsz, meas, useabs, sdim, eps, minel = cfg.values
            ds.classification = "temporal classif w/ %r" % (cfg.values,)
            self.status.emit(ds.classification)
            ys = ds.data if True else ds.data[ds.pick_mask]
            peaks = [e["time"] for e in ds.events]
            echan = [e["targets"][0] for e in ds.events]
            ds.oob, ds.epochs = util.extract_windows(
                ds.fs, ys, peaks, pre=-winsz / 2.0, post=winsz / 2.0, echan=echan
            )
            self.status.emit("windows extracted, computing similarity..")
            if meas == "corrcoef":
                ds.ss = np.corrcoef(ds.epochs)
            elif meas == "cov":
                ds.ss = np.cov(ds.epochs)
            else:
                Exception("incorrect measure %r" % (meas,))
            if useabs:
                ds.ss = np.abs(ds.ss)
            ds.ss = (ds.ss - ds.ss.min()) / ds.ss.ptp()
            self.status.emit("computing spectral embedding & clustering")
            ds.xi, ds.evlabels = cluster.spectral_dbscan(
                ds.ss, n_dim=sdim, eps=eps, min_samples=minel
            )
            for e, o, el in zip(ds.events, ds.oob, ds.evlabels):
                e["value"] = 4199 if o else 4201 + el


class ClassSpace(ParAction):
    _text = "&Spatial classification"
    _shortcut = "F6"

    def run(self):
        cfg = ParamsDialog(
            parent=self.main,
            parameters=[
                {
                    "name": "Window size (pre + post)",
                    "type": "float",
                    "value": 0.4,
                    "siPrefix": True,
                    "suffix": "s",
                    "step": 0.05,
                },
                {
                    "name": "Use all channels\n(use picked if unchecked)",
                    "type": "bool",
                    "value": False,
                },
                {
                    "name": "N. spatial components",
                    "type": "int",
                    "value": 3,
                    "limits": (1, 1000),
                },
                {
                    "name": "Similarity dimension",
                    "type": "int",
                    "value": 2,
                    "limits": (1, 3),
                },
                {
                    "name": "Neighborhood quantile",
                    "type": "float",
                    "value": 0.05,
                    "limits": (0.0, 1.0),
                    "step": 0.01,
                },
                {
                    "name": "Min. n. class elements",
                    "type": "int",
                    "value": 20,
                    "limits": (2, 1000000),
                    "step": 10,
                },
            ],
        )
        ret = cfg.exec_()
        if ret:
            ds = self.main.dataset
            winsz, useall, ncomp, sdim, eps, minsamp = cfg.values
            ds.classification = "spatial classif w/ %r" % (cfg.values,)
            self.status.emit(ds.classification)
            ys = ds.data if useall else ds.data[ds.pick_mask]
            peaks = [e["time"] for e in ds.events]
            ds.oob, ds.epochs = util.extract_windows(
                ds.fs, ys, peaks, pre=-winsz / 2.0, post=winsz / 2.0
            )
            self.status.emit("windows extracted, computing similarity..")
            ds.us, ds.ss = cluster.subspace_similarity(ds.epochs, ncomp)
            self.status.emit("computing spectral embedding & clustering")
            ds.xi, ds.evlabels = cluster.spectral_dbscan(
                ds.ss, n_dim=sdim, eps=eps, min_samples=minsamp
            )
            for e, o, el in zip(ds.events, ds.oob, ds.evlabels):
                e["value"] = 4199 if o else 4201 + el


actions = [ClassSpace, ClassTime, "separator", ExtractEvents, EventSimilarity, Cluster]
