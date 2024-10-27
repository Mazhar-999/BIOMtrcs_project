import os
import librosa
import numpy as np
import joblib

def extract_features_from_audio(file_path):
    """Load an audio file and extract its MFCC features."""
    try:
        # Load the audio file
        signal, sr = librosa.load(file_path, sr=None)
        
        # Normalize the audio signal
        signal = signal / np.max(np.abs(signal))  # Normalize the signal
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)  # Use 'y' for signal and 'sr' for sample rate
        
        # Calculate the mean of the MFCC features
        mfccs_mean = np.mean(mfccs, axis=1)
        
        return mfccs_mean

    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None  # Return None if there's an error

def load_data(authorized_dir, unauthorized_dir):
    """Load audio data and extract features."""
    features = []
    labels = []

    # Load authorized data
    for filename in os.listdir(os.path.join(authorized_dir, 'voice_samples')):
        if filename.endswith('.wav'):
            audio_path = os.path.join(authorized_dir, 'voice_samples', filename)
            feature = extract_features_from_audio(audio_path)
            if feature is not None:  # Ensure feature is not None
                features.append(feature)
                labels.append(1)  # 1 for authorized

    # Load unauthorized data
    for filename in os.listdir(os.path.join(unauthorized_dir, 'voice_samples')):
        if filename.endswith('.wav'):
            audio_path = os.path.join(unauthorized_dir, 'voice_samples', filename)
            feature = extract_features_from_audio(audio_path)
            if feature is not None:  # Ensure feature is not None
                features.append(feature)
                labels.append(0)  # 0 for unauthorized

    return np.array(features), np.array(labels)

def main():
    # Set paths for authorized and unauthorized data
    authorized_dir = "data/authorized/"
    unauthorized_dir = "data/unauthorized/"
    
    # Create features directories if they don't exist
    os.makedirs(os.path.join(authorized_dir, 'features'), exist_ok=True)
    os.makedirs(os.path.join(unauthorized_dir, 'features'), exist_ok=True)

    # Load data and extract features
    features, labels = load_data(authorized_dir, unauthorized_dir)

    if len(features) == 0 or len(labels) == 0:
        print("No features or labels loaded. Exiting.")
        return

    # Save features and labels using joblib
    joblib.dump(features, os.path.join(authorized_dir, 'features', 'features.pkl'))
    joblib.dump(labels, os.path.join(unauthorized_dir, 'features', 'labels.pkl'))

if __name__ == "__main__":
    main()

def load_your_data():
    """
    Loads features and labels for training.

    Returns:
    - X: 2D array of features
    - y: 1D array of labels
    """
    X = []
    y = []

    # Define the directories for authorized and unauthorized voice samples
    for label in ['authorized', 'unauthorized']:
        # Path to the audio samples for each label
        audio_dir = f'data/{label}/voice_samples/'

        # Iterate over each audio file in the directory
        for audio_file in os.listdir(audio_dir):
            audio_path = os.path.join(audio_dir, audio_file)

            # Extract features from the audio file
            features = extract_features_from_audio(audio_path)

            # Check if features were successfully extracted
            if features is not None and features.size > 0:
                X.append(features)  # Append the features to the list
                # Assign label: 1 for authorized, 0 for unauthorized
                y.append(1 if label == 'authorized' else 0)  

    # Convert lists to numpy arrays
    X = np.vstack(X) if X else np.empty((0, 0))  # Stack features into a 2D array
    y = np.array(y)  # Convert labels to a numpy array

    # Print shapes for debugging
    print("Loaded features shape:", X.shape)
    print("Loaded labels shape:", y.shape)

    return X, y  # Return the features and labels
