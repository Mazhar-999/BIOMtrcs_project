import os
import librosa
import numpy as np
import joblib

def extract_features_from_audio(file_path):
    # Load the audio file
    signal, sr = librosa.load(file_path, sr=None)
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    
    # Calculate the mean of the MFCC features
    mfccs_mean = np.mean(mfccs, axis=1)
    
    return mfccs_mean

def predict_audio(model, audio_path):
    # Extract features from the audio sample
    features = extract_features_from_audio(audio_path)
    
    # Reshape the features to match the model input
    features = features.reshape(1, -1)  # Reshape for a single sample
    
    # Make a prediction
    prediction = model.predict(features)
    
    return prediction[0]



def main():
    # Load the trained model
    model = joblib.load('voice_authentication_model.pkl')
    
    # Set the path to the audio file you want to test
    test_audio_path = 'data/test_sample.wav'  # Change this to your test file path
    
    # Make a prediction
    result = predict_audio(model, test_audio_path)
    
    if result == 1:
        print("Access Granted: Authorized User")
    else:
        print("Access Denied: Unauthorized User")

if __name__ == "__main__":
    main()
