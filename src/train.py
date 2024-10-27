import os
import joblib
import numpy as np
import librosa  # Import librosa for audio processing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_model():
    # Define your directory paths here
    authorized_dir = 'data/authorized/voice_samples'
    unauthorized_dir = 'data/unauthorized/voice_samples'

    features, labels = load_data(authorized_dir, unauthorized_dir)

    # Debug: Print shapes of features and labels
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    model = RandomForestClassifier()  # Or whichever model you're using
    model.fit(features, labels)
    
    # Save the trained model
    joblib.dump(model, 'models/voice_auth_model.pkl')



def extract_features_from_audio(file_path):
    """Extract MFCC features from an audio file."""
    # Load the audio file
    signal, sr = librosa.load(file_path, sr=None)
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)  # Use 'y' for signal and 'sr' for sample rate
    
    # Calculate the mean of the MFCC features
    mfccs_mean = np.mean(mfccs, axis=1)
    
    return mfccs_mean

def load_data(authorized_dir, unauthorized_dir):
    features = []
    labels = []
    
    # Process authorized samples
    for audio_file in os.listdir(authorized_dir):
        audio_path = os.path.join(authorized_dir, audio_file)
        feature = extract_features_from_audio(audio_path)
        features.append(feature)
        labels.append(1)  # Assuming 1 for authorized
    
    # Process unauthorized samples
    for audio_file in os.listdir(unauthorized_dir):
        audio_path = os.path.join(unauthorized_dir, audio_file)
        feature = extract_features_from_audio(audio_path)
        features.append(feature)
        labels.append(0)  # Assuming 0 for unauthorized

    return np.array(features), np.array(labels)


def main():
    # Set paths for authorized and unauthorized data
    authorized_dir = "data/authorized/"
    unauthorized_dir = "data/unauthorized/"

    # Load data and extract features
    features, labels = load_data(authorized_dir, unauthorized_dir)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Create a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save the trained model
    joblib.dump(model, 'data/model.pkl')
  
