# Usage

Run IC/DC by double clicking run.bat (Windows), or running `python -m icdc` in the console (Mac/Linux).

# Dependencies

- Python 2.7, NumPy, Scipy
- PyQt/PySide, pyqtgraph
- pyopengl
- PyDICOM
- PyWavelets
- sklearn

# Pending dependences

- docopt for command line parsing
- mne python

# Conda environment

You can use conda to easily create an environment for using ICDC,
```
conda create -n icdc numpy scipy pyqt pyqtgraph pyopengl PyWavelets scikit-learn
conda activate icdc
pip install dicom mne
```
