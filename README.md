# Steps

Hey yo! this is how I trained that mtf

Before starting we need to install things and make an environment.

```
python -m venv ser_env
source ser_env/bin/activate
pip install torch torchaudio transformers datasets librosa scikit-learn pandas numpy
```

Make sure you have this file anywhere and extract it anywhere.
```
https://zenodo.org/records/1188976/files/Audio_Song_Actors_01-24.zip?download=1
```
Just note that you need to change this line in 1.py base on where you extracted the data sets okay?
```
ravdess_df = load_ravdess_data("/home/polo/SER/test-ser/ravdess")
```
Run things in sequence!
```
python 1.py
python 2.py
python 3.py
```

You might need to do some modifications in 3.py but its self explanatory, Even I understood it so you can do it too!
