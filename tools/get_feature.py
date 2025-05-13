import librosa



def collect_features(y, sr):
    avg_cqt_chroma, _ = chroma_cqt(y, sr)
    features = {
        'tempo': tempo(y, sr),
        'spectral_centroid': spectral_centroid(y, sr),
        'zero_crossing_rate': zero_crossing_rate(y),
        'spectral_rolloff': spectral_rolloff(y, sr),
        'chroma_cqt': avg_cqt_chroma,
        'chroma_cens': chroma_cens(y, sr),
        'mfcc': mfcc(y, sr),
    }
    return features


def tempo(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return tempo[0]


def spectral_centroid(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return centroid.mean()


def zero_crossing_rate(y):
    zcr = librosa.feature.zero_crossing_rate(y)
    return zcr.mean()


def spectral_rolloff(y, sr):
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95)
    return rolloff.mean()


def chroma_cqt(y, sr):
    cqt_chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    return cqt_chroma.mean(axis=1), cqt_chroma


def chroma_cens(y, sr):
    _, cqt_chroma = chroma_cqt(y, sr)
    chroma = librosa.feature.chroma_cens(C=cqt_chroma) 
    return chroma.mean(axis=1) 

def mfcc(y, sr, n_mfcc=30):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs.mean(axis=1)