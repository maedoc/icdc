# Usage

You can use conda to easily create an environment for using ICDC,
```
conda create -n icdc numpy scipy pyqt pyqtgraph pyopengl PyWavelets scikit-learn
conda activate icdc
pip install dicom mne
```

Then run it with 
```
python -m icdc
```