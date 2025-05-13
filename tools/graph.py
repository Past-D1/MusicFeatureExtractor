import librosa, librosa.display
import matplotlib.pyplot as plt



def spectral_centroid(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    plt.figure(figsize=(10, 6))
    plt.semilogy(centroid.T, label='Spectral Centroid')
    plt.ylabel('Spectral Centroid (Hz)')
    plt.xticks([])
    plt.xlim([0, centroid.shape[-1]])
    plt.title('Spectral Centroid')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def zero_crossing_rate(y):
    zcr = librosa.feature.zero_crossing_rate(y)
    plt.figure(figsize=(10, 6))
    plt.plot(zcr.T, label='Zero Crossing Rate')
    plt.ylabel('Zero Crossing Rate')
    plt.xticks([])
    plt.xlim([0, zcr.shape[-1]])
    plt.title('Zero Crossing Rate')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def spectral_rolloff(y, sr):
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95)
    plt.figure(figsize=(10, 6))
    plt.semilogy(rolloff.T, label='Spectral Rolloff (95%)')
    plt.ylabel('Spectral Rolloff (Hz)')
    plt.xticks([])
    plt.xlim([0, rolloff.shape[-1]])
    plt.title('Spectral Rolloff')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def mfcc(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time', sr=sr)
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()