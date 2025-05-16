from tools import graph
from tools.audio_utils import Audio


# This script is designed to analyze audio files and extract features such as:
# tempo, spectral centroid, zero crossing rate, spectral rolloff, chroma features, and MFCCs (with more to come).
# It also provides options to visualize some of these features using matplotlib.

# I have every intention of stepping away from the current command line interface (CLI) and
# moving towards a more user-friendly GUI in the future.


def main():
    while True:
        print("Welcome to the Music Feature Explorer!")
        filepath = input("Please enter the path to an audio file (e.g., 'audio.mp3' or 'audio.wav'): ").strip()


        audio = Audio.load(filepath)
        if audio is None:
            return
        
        print("Extracting features...")
        features = audio.extract.all()
        print("Feature extraction completed.")
        audio.print_features.all()

        print("Feature extraction completed.")
        while True:
            shown_features = input("Would you like to visualize the features? (y/n): ").strip()
            if shown_features not in ['y', 'Y', 'n', 'N']:
                print("Invalid input. Please enter 'y' or 'n'.")
                continue
            elif shown_features == 'n' or 'N':
                print("Thank you for using the Music Feature Explorer!")
                break
            else:
                print("Feature visualization is currently disabled.")
                #print("Visualizing features...")
                #graph.spectral_centroid(y, sr)
                #graph.zero_crossing_rate(y)
                #graph.spectral_rolloff(y, sr)
                #graph.mfcc(y, sr)
                #print("Visualization completed.")
                break

        while True:
            another = input("Would you like to analyze another audio file? (y/n): ").strip()
            if another not in ['y', 'Y', 'n', 'N']:
                print("Invalid input. Please enter 'y' or 'n'.")
                continue
            elif another == 'y' or 'Y':
                break
            else:
                print("Thank you for using the Music Feature Explorer!")
                return




if __name__ == '__main__':
    main()
