import librosa, os
from tools import graph, get_feature


def main():
    while True:
        print("Welcome to the Music Feature Explorer!")
        file_path = input("Enter the path to an audio file (e.g., 'audio.mp3' or 'audio.wav'): ").strip()

        if not os.path.isfile(file_path):
            print("File does not exist. PLease check the path and try again.")
            return

        y, sr = load_audio(file_path)
        if y is None:
            return
        
        print("Extracting features...")
        features = get_feature.collect_features(y, sr)
        print(f"""
Feature Extraction Results:
Estimated Tempo: {features['tempo']:.2f} BPM
Average Spectral Centroid: {features['spectral_centroid']:.2f} Hz
Average Zero Crossing Rate: {features['zero_crossing_rate']:.4f}
Average Spectral Rolloff: {features['spectral_rolloff']:.2f} Hz
Average Chroma CQT (per pitch class):""")
        for i, val in enumerate(features['chroma_cqt']):
            print(f"  Pitch class {i}: {val:.4f}")
            if i == len(features['chroma_cqt']) - 1:
                print(f"Average Chroma CENS (per pitch class):")
                for j, val in enumerate(features['chroma_cens']):
                    print(f"  Pitch class {j}: {val:.4f}")
                    if j == len(features['chroma_cens']) - 1:
                        print(f"Average MFCC (first 30 coefficients):")
                        for k, val in enumerate(features['mfcc']):
                            print(f"  Coefficient {k}: {val:.4f}")

        print("Feature extraction completed.")
        while True:
            shown_features = input("Would you like to visualize the features? (y/n): ").strip().lower()
            if shown_features not in ['y', 'n']:
                print("Invalid input. Please enter 'y' or 'n'.")
                continue
            elif shown_features != 'y':
                print("Thank you for using the Music Feature Explorer!")
                break
            else:
                print("Visualizing features...")
                graph.spectral_centroid(y, sr)
                graph.zero_crossing_rate(y)
                graph.spectral_rolloff(y, sr)
                graph.mfcc(y, sr)
                print("Visualization completed.")
                break

        while True:
            another = input("Would you like to analyze another audio file? (y/n): ").strip().lower()
            if another not in ['y', 'n']:
                print("Invalid input. Please enter 'y' or 'n'.")
                continue
            elif another == 'y':
                break
            else:
                print("Thank you for using the Music Feature Explorer!")
                return


def load_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        print(f"Audio loaded successfully. Sample rate: {sr}, Duration: {len(y) / sr:.2f} seconds")
        return y, sr
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None, None


if __name__ == '__main__':
    main()