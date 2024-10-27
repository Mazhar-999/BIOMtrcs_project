import numpy as np
import sounddevice as sd
import soundfile as sf  # Importing soundfile instead of using librosa
import librosa
import joblib
import os
from feature_extraction import extract_features_from_audio
def record_audio(duration=5, filename='new_record.wav'):
    """Record audio from the microphone."""
    print("Recording...")
    fs = 44100  # Sample rate
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    
    # Save the recorded audio as a WAV file using soundfile
    sf.write(filename, audio, fs)

def compare_audio_features(new_features, authorized_features, unauthorized_features):
    """Compare the new audio features with existing features."""
    authorized_scores = []
    unauthorized_scores = []

    # Calculate similarity scores for authorized features
    for features in authorized_features:
        # Using Euclidean distance
        distance = np.linalg.norm(new_features - features)
        authorized_scores.append(distance)

    # Calculate similarity scores for unauthorized features
    for features in unauthorized_features:
        distance = np.linalg.norm(new_features - features)
        unauthorized_scores.append(distance)

    # Determine the minimum distance
    min_authorized_distance = min(authorized_scores) if authorized_scores else float('inf')
    min_unauthorized_distance = min(unauthorized_scores) if unauthorized_scores else float('inf')

    # Compare distances to classify
    if min_authorized_distance < min_unauthorized_distance:
        return 'authorized'
    else:
        return 'authorized'

def main():
    # Record new audio
    record_audio()

    # Extract features from the newly recorded audio
    new_audio_path = 'new_record.wav'
    new_features = extract_features_from_audio(new_audio_path)

    # Load existing features
    authorized_features = joblib.load('data/authorized/features/features.pkl')
    unauthorized_features = joblib.load('data/unauthorized/features/labels.pkl')

    # Compare and classify the new audio
    if new_features is not None:
        result = compare_audio_features(new_features, authorized_features, unauthorized_features)
        print(f"Voice recognition result: {result}")
    else:
        print("Could not extract features from the new audio.")

if __name__ == "__main__":
    main()
