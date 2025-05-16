import librosa, numpy as np

class Audio:
    def __init__(self, y: np.ndarray = None, sr: int = None, filepath: str = None, return_series=False):
        self.y = y
        self.sr = sr
        self.filepath = filepath
        self.return_series = return_series
        self.extract = self.FeatureExtractor(self)
        self.print_features = self.PrintFeatures(self)

    @classmethod
    def load(cls, filepath):
        import os
        if not os.path.isfile(filepath):
            print(f"File does not exist: {filepath}")
            return
        
        try:
            y, sr = librosa.load(filepath, sr=22050)
            duration = librosa.get_duration(y=y, sr=sr)
            print(f"Successfully loaded {filepath}. Sample rate: {sr} Duration {duration:.2f} seconds.")
            return cls(y=y, sr=sr, filepath=filepath)
        except Exception as e:
            print(f"Error loading audio file {filepath}: {e}")
            return None, None
    
    
    
    class PrintFeatures:
        def __init__(self, parent):
            self.audio = parent
            self.extract = parent.extract

        def __call__(self):
            raise NotImplementedError("Call a specific feature extractor method instead.")
        
        def all(self):
            features = self.extract.all()
            if features['low_level']:
                print("Low-level features:")
                for k, v in features['low_level'].items():
                    print(f"  {k}: {v}")
            else:
                print("  No low-level features found.")
            if features['mid_level']:
                print("Mid-level features:")
                for k, v in features['mid_level'].items():
                    print(f"  {k}: {v}")
            else:
                print("  No mid-level features found.")
            if features['high_level']:
                print("High-level features:")
                for k, v in features['high_level'].items():
                    print(f"  {k}: {v}")
            else:
                print("  No high-level features found.")

        def low(self):
            features = self.extract.low()
            if features:
                print("Low-level features:")
                for k, v in features['low_level'].items():
                    print(f"  {k}: {v}")
            else:
                print("  No low-level features found.")

        def mid(self):
            features = self.extract.mid()
            if features:
                print("Mid-level features:")
                for k, v in features['mid_level'].items():
                    print(f"  {k}: {v}")
            else:
                print("  No mid-level features found.")

        def high(self):
            features = self.extract.high()
            if features:
                print("High-level features:")
                for k, v in features['high_level'].items():
                    print(f"  {k}: {v}")
            else:
                print("  No high-level features found.")
    
    
    class FeatureExtractor:
        def __init__(self, parent):
            self.audio = parent

        def __call__(self):
            raise NotImplementedError("Call a specific feature extractor method instead.")
        
        def all(self):
            return {
                'low_level': self.low(),
                'mid_level': self.mid(),
                'high_level': self.high(),
            }

        def low(self):
            return low_level_features(self.audio.y, self.audio.sr, return_series=self.audio.return_series)

        def mid(self):
            return None

        def high(self):
            return None
    




# Utility functions for feature extraction

def low_level_features(y, sr): # creates a dictionary for the low-level features of the audio
    chroma_cqt_list, _ = chroma_cqt(y, sr)
    tempo, _ = tempo_and_beats(y, sr)

    ll_features = {
        'tempo': tempo,
        'spectral_centroid': spectral_centroid(y, sr),
        'zero_crossing_rate': zero_crossing_rate(y),
        'spectral_rolloff': spectral_rolloff(y, sr),
        'chroma_cqt': chroma_cqt_list,
        'chroma_cens': chroma_cens(y, sr),
        'mfcc': mfcc(y, sr),
    }
    return ll_features # the "ll" stands for "low-level" features


def tempo_and_beats(y, sr):
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    return tempo[0], beats


def spectral_centroid(y, sr, return_series=False):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    sc_list = {
        'mean': centroid.mean(),
        'std': centroid.std(),
        'min': centroid.min(),
        'max': centroid.max(),
    }
    if return_series:
        sc_list['series'] = centroid.tolist()
    return sc_list


def zero_crossing_rate(y, return_series=False):
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_list = {
        'mean': zcr.mean(),
        'std': zcr.std(),
        'min': zcr.min(),
        'max': zcr.max(),
    }
    if return_series:
        zcr_list['series'] = zcr.tolist()
    return zcr_list


def spectral_rolloff(y, sr, return_series=False):
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95)
    rolloff_list = {
        'mean': rolloff.mean(),
        'std': rolloff.std(),
        'min': rolloff.min(),
        'max': rolloff.max(),
    }
    if return_series:
        rolloff_list['series'] = rolloff.tolist()
    return rolloff_list


def chroma_cqt(y, sr, return_series=False):
    cqt_chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cqt_list = {
        'mean': cqt_chroma.mean(axis=1),
        'std': cqt_chroma.std(axis=1),
        'min': cqt_chroma.min(axis=1),
        'max': cqt_chroma.max(axis=1),
    }
    if return_series:
        chroma_cqt_list['series'] = cqt_chroma.tolist()
    return chroma_cqt_list, cqt_chroma

def chroma_cens(y, sr, return_series=False):
    _, cqt_chroma = chroma_cqt(y, sr)
    chroma = librosa.feature.chroma_cens(C=cqt_chroma) 
    chroma_cens_list = {
        'mean': chroma.mean(axis=1),
        'std': chroma.std(axis=1),
        'min': chroma.min(axis=1),
        'max': chroma.max(axis=1),
    }
    if return_series:
        chroma_cens_list['series'] = chroma.tolist()
    return chroma_cens_list

def mfcc(y, sr, n_mfcc=30, return_series=False):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_list = {
        'mean': mfccs.mean(axis=1),
        'std': mfccs.std(axis=1),
        'min': mfccs.min(axis=1),
        'max': mfccs.max(axis=1),
    }
    if return_series:
        mfccs_list['series'] = mfccs.tolist()
    return mfccs_list
