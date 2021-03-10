import os
import io
import configparser

import numpy as np
from scipy.io import loadmat
import mne

from .core import Dataset


class VHDR(Dataset):
    """
    Brain Vision file.

    """

    def __init__(self, filename, **readkwds):
        super(VHDR, self).__init__(filename)
        self.wd, _ = os.path.split(filename)

        # read file
        with open(self.filename, "r") as fd:
            self.srclines = [_.decode('ascii', 'ignore') for _ in fd.readlines()]

        # config parser expects each section to have header
        # but vhdr has some decorative information at the beginning
        while not self.srclines[0].startswith("["):
            self.srclines.pop(0)

        self.sio = io.StringIO()
        self.sio.write("\n".join(self.srclines))
        self.sio.seek(0)

        self.cp = configparser.ConfigParser()
        self.cp.readfp(self.sio)

        for opt in self.cp.options("Common Infos"):
            setattr(self, opt, self.cp.get("Common Infos", opt))

        self.binaryformat = self.cp.get("Binary Infos", "BinaryFormat")

        self.labels = [
            self.cp.get("Channel Infos", o).split(",")[0]
            for o in self.cp.options("Channel Infos")
        ]

        self.fs = self.srate = 1e6 / float(self.samplinginterval)
        self.nchan = int(self.numberofchannels)

        # important if not in same directory
        self.datafile = os.path.join(self.wd, self.datafile)

        self.read_data(**readkwds)

    def read_data(self, mmap=False, dt="float32", mode="r"):
        """
        VHDR stores data in channel contiguous way such that reading disparate pieces
        in time is fast, when using memmap.
        """

        if mmap:
            ary = np.memmap(self.datafile, dt, mode)
        else:
            ary = np.fromfile(self.datafile, dt)
        self.data = ary.reshape((-1, self.nchan)).T
        self.nsamp = self.data.shape[1]


class EEGLAB(Dataset):
    "EEGLAB .set file"

    def __init__(self, filename):
        super(EEGLAB, self).__init__(filename)
        self.mat = loadmat(filename)
        self.fs = self.mat["EEG"]["srate"][0, 0][0, 0]
        self.nsamp = self.mat["EEG"]["pnts"][0, 0][0, 0]
        self.data = np.fromfile(
            ".".join(filename.split(".")[:-1]) + ".fdt", dtype=np.float32
        )
        self.data = self.data.reshape((self.nsamp, -1)).T
        self.nchan = self.data.shape[0]
        self.labels = [c[0] for c in self.mat["EEG"]["chanlocs"][0, 0]["labels"][0]]


class MATFile(Dataset):
    def __init__(self, filename):
        super(MATFile, self).__init__(filename)
        self.mat = loadmat(filename)
        self.fs = self.mat["fs"][0, 0] * 1.0
        self.data = self.mat["data"]
        self.nsamp = self.data.shape[1]
        self.nchan = self.data.shape[0]
        self.labels = [l[0] for l in self.mat["labels"][0]]


class NPZFile(Dataset):
    def __init__(self, filename):
        super(NPZFile, self).__init__(filename)
        z = np.load(filename)
        self.fs = z["fs"].flat[0]
        self.data = z["data"]
        self.nsamp = self.data.shape[1]
        self.nchan = self.data.shape[0]
        self.labels = list(z["labels"])


# MNE has a bunch of file readers, MNEReader is a base class,

class MNEReader:
    def __init__(self, filename):
        super().__init__(filename)
        raw = self._read_raw(filename)
        data, times = raw[:]
        self.fs = 1.0 / (times[1] - times[0])
        self.data = data
        self.nsamp = len(times)
        self.labels = raw.info['ch_names']
        self.nchan = len(self.labels)

# then just add a new class per read function 
# cf https://mne.tools/stable/search.html?q=read_raw to see 
# the other functions gdf, ctf, bti, etc

class EdfFile(Dataset, MNEReader):
    _read_raw = mne.io.read_raw_edf


class FifFile(Dataset, MNEReader):
    _read_raw = mne.io.read_raw_fif


class MarkersCSV(object):
    pass


"""
try:
    import openpyxl

    class XLS(object):
        # TODO useful to export data in this format
        def read(filename):
            wb = openpyxl.load_workbook(filename)
            sheets = []
            for sheet in wb.worksheets:
                values = [[cell.value for cell in col] for col in sheet.columns]
                array = numpy.array(values, object).T
                sheets.append(array)
            return sheets

        def write(filename):
            pass
"""
