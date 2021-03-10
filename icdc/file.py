# If you add actions, remember to register at the bottom of the file!

import os

import numpy as np
from scipy.io import savemat

try:
    import pickle as pickle
except ImportError:
    import pickle

from .qt import QtCore, QtGui
from .core import Action
from .fileio import EEGLAB, VHDR, MATFile, NPZFile


# TODO refactor a settings class to handle this generally
def read_recent_files(qs):
    fs = []
    for i in range(qs.beginReadArray("recent-files")):
        qs.setArrayIndex(i)
        name = qs.value("name")
        try:
            name = name.toString()
        except:
            pass
        fs.append(str(name))
    qs.endArray()
    return fs


def write_recent_files(qs, fs):
    qs.beginWriteArray("recent-files")
    for i, f in enumerate(fs):
        qs.setArrayIndex(i)
        qs.setValue("name", f)
    qs.endArray()


def add_recent_file(qs, f):
    rf = read_recent_files(qs)
    if f in rf:
        rf.remove(f)
    rf.append(f)
    if len(rf) > 10:
        rf = rf[:-10]
    write_recent_files(qs, rf)


def find_file(filt="Any files (*.*)", parent=None, mode="open"):
    "Common interface to ask user for a file location"

    # get settings for our app
    qs = QtCore.QSettings("INS", "IC/DC")

    # try to get string from settings
    path = qs.value("last-dir")
    if type(path) not in (str, str):
        try:
            path = str(path.toString())
        except:
            pass

    # otherwise use home directory
    if path is None or len(path) == 0:
        path = os.path.expanduser("~")

    if mode == "open":
        dlg = QtGui.QFileDialog.getOpenFileName
    elif mode == "save":
        dlg = QtGui.QFileDialog.getSaveFileName
    else:
        raise Exception("unknown find file dialog mode %r" % (mode,))

    newpath = dlg(parent, "Select file", path, filt)

    if isinstance(newpath, tuple):
        newpath, filt_used = newpath

    if len(newpath) > 0:
        path = str(newpath)
        if mode == "open":
            add_recent_file(qs, path)
        qs.setValue("last-dir", os.path.dirname(path))
        return path
    else:
        # cancel!
        return None


class NewWindow(Action):
    _text = "&Open new window"
    _shortcut = "Ctrl+N"

    def run(self):
        from .main import Main

        Main().show()


class CloneWindow(Action):
    _text = "Open new window with current &data"
    _shortcut = "Ctrl+Shift+N"

    def run(self):
        self.main.clone_window()


class LoadEEGLAB(Action):
    _text = "Load an EEGLAB dataset (.&set)"

    def run(self, filename=None):
        filename = filename or find_file("EEGLAB datasets (*.set)", parent=self.main)
        if filename:
            self.status.emit("reading %s" % (filename,))
            self.main.dataset.update(EEGLAB(filename))
            self.status.emit("done!")


class LoadVHDR(Action):
    _text = "Load a Brain Vision dataset (.&vhdr)"

    def run(self, filename=None):
        filename = filename or find_file(
            "BrainVision headers (*.vhdr)", parent=self.main
        )
        if filename:
            self.status.emit("reading %s" % (filename,))
            self.main.dataset.update(VHDR(filename))
            self.status.emit("done!")


class LoadMATFile(Action):
    _text = "Load a MATLAB file (.&mat)"

    def run(self, filename=None):
        filename = filename or find_file(filt="MATLAB file (*.mat)", parent=self.main)
        if filename:
            self.status.emit("reading %s" % (filename,))
            self.main.dataset.update(MATFile(filename))
            self.status.emit("done!")


class LoadAWMarkers(Action):
    _text = "Load an AnyWave marker file (.m&rk)"

    def run(self, filename=None):
        filename = filename or find_file(
            filt="AnyWave marker files (*.mrk)", parent=self.main
        )
        if filename:
            ds = self.main.dataset
            self.status.emit("reading events from %s" % (filename,))
            with open(filename, "r") as fd:
                for line in fd.readlines():
                    if line.startswith("//"):
                        continue
                    parts = line.split("\t")
                    self.status.emit(line)
                    if len(parts) < 4:  # no dur specified
                        parts.append("0.0")
                    if len(parts) < 5:  # no targets specified
                        parts.append("")
                    ds.events.append(
                        {
                            "label": parts[0],
                            "value": int(parts[1]),
                            "time": float(parts[2]),
                            "duration": float(parts[3]),
                            "targets": [
                                t for t in parts[4].strip().split(",") if len(t) > 0
                            ],
                        }
                    )


class SaveAWMarkers(Action):
    _text = "Save an AnyWave marker file (.m&rk)"

    def run(self):
        filename = find_file(
            filt="AnyWave marker files (*.mrk)", parent=self.main, mode="save"
        )
        if filename:
            ds = self.main.dataset
            self.status.emit("writing events to %s" % (filename,))
            with open(filename, "w") as fd:
                fd.write("// AnyWave marker file\n")
                for e in ds.events:
                    val = e.get("value", 4200)
                    dur = e.get("duration", 0.0)
                    targets = e.get("targets", [])
                    line = "%s\t%d\t%.3f\t%f\t%s\n"
                    if targets and (type(targets[0]) not in (str, str)):
                        targets = [ds.labels[i] for i in targets]
                    line %= (e["label"], val, e["time"], dur, ",".join(targets))
                    fd.write(line)
            self.status.emit("done!")
        else:
            self.status.emit("save canceled!")


class SaveEventsCsv(Action):
    _text = "Save events as comma-separated values (.&csv)"


class SaveEventsMAT(Action):
    _text = "Save detections in a MATLAB file (.&mat)"

    def run(self, filename=None):
        filename = filename or find_file(
            filt="MATLAB file (*.mat)", parent=self.main, mode="save"
        )
        if filename:
            es = self.main.dataset.events
            ts = np.array([e["time"] for e in es])
            vs = np.array([e["value"] for e in es])
            cs = np.array([e["targets"] for e in es])
            savemat(filename, {"times": ts, "values": vs, "chanids": cs})


class SaveWorkspace(Action):
    _text = "Save &workspace"
    _shortcut = "Ctrl+S"

    def run(self, filename=None):
        filename = filename or find_file(
            filt="Workspace pickles (*.pickle)", parent=self.main, mode="save"
        )
        if filename:
            self.status.emit("pickling workspace...")
            with open(filename, "w") as fd:
                pickle.dump(self.main.dataset, fd)


class LoadWorkspace(Action):
    _text = "Load &workspace"
    _shortcut = "Ctrl+O"

    def run(self, filename=None):
        filename = filename or find_file(
            filt="Workspace pickles (*.pickle)", parent=self.main
        )
        if filename:
            self.status.emit("unpickling workspace...")
            with open(filename, "r") as fd:
                self.main.dataset = pickle.load(fd)
            self.main.sig_data_changed.emit(self.main.dataset)


class SaveEEGLAB(Action):
    _text = "Save data as EEGLAB dataset (.&set)"


class SaveVHDR(Action):
    _text = "Save data in Brain Vision format (.&vhdr)"


class LoadNPZ(Action):
    _text = "Load data from NumPy &ZIP (.npz)"

    def run(self, filename=None):
        filename = filename or find_file(
            filt="NumPy ZIP file (*.npz)", parent=self.main
        )
        if filename:
            self.status.emit("reading %s" % (filename,))
            self.main.dataset.update(NPZFile(filename))


class SaveNPZ(Action):
    _text = "Save data to NumPy &ZIP (.npz)"

    def run(self, filename=None):
        filename = filename or find_file(
            filt="NumPy ZIP file (*.npz)", parent=self.main, mode="save"
        )
        if filename:
            ds = self.main.dataset
            self.status.emit("saving to %s" % (filename,))
            np.savez(filename, data=ds.data, fs=ds.fs, labels=ds.labels)


class RecentFileAction(Action):
    def __init__(self, idx, filename, parent=None):
        self.filename = filename
        try:
            self.filename = str(self.filename.toString())
        except Exception as exc:
            print(exc)
        self._text = "&%d %s" % (idx, filename)
        Action.__init__(self, parent=parent)

    def run(self):
        # TODO can just do Action(parent).run(filename) now
        add_recent_file(QtCore.QSettings("INS", "IC/DC"), self.filename)
        if self._text.endswith(".set"):
            ds = EEGLAB(self.filename)
            self.main.dataset.update(ds)
        elif self._text.endswith(".vhdr"):
            ds = VHDR(self.filename)
            self.main.dataset.update(ds)
        elif self._text.endswith(".mat"):
            LoadMATFile(parent=self.parentWidget()).run(self.filename)
        elif self._text.endswith(".mrk"):
            LoadAWMarkers(parent=self.parentWidget()).run(self.filename)
        elif self.filename.endswith("pickle"):
            LoadWorkspace(parent=self.parentWidget()).run(self.filename)
        else:
            raise Exception(
                "do not know how to load a %r file" % (self.filename.split(".")[-1],)
            )


class ClearRecentFiles(Action):
    _text = "Clear recent files"

    def run(self):
        write_recent_files(QtCore.QSettings("INS", "IC/DC"), [])


title = "&File"
actions = [
    NewWindow,
    CloneWindow,
    "separator",
    LoadWorkspace,
    LoadEEGLAB,
    LoadVHDR,
    LoadMATFile,
    LoadAWMarkers,
    LoadNPZ,
    "separator",
    SaveWorkspace,
    SaveAWMarkers,
    SaveEventsCsv,
    SaveEventsMAT,
    SaveEEGLAB,
    SaveVHDR,
    SaveNPZ,
    "separator",
]
