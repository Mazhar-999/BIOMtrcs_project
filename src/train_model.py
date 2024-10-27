import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.feature_extraction import load_your_data  # Ensure this function is defined in feature_extraction.py

def train_model():
    """
    Trains the voice authentication model using extracted features and labels.
    """
    try:
        X, y = load_your_data()  # Load your training data
        
        # Check shapes before training
        print("Features shape before training:", X.shape)
        print("Labels shape before training:", y.shape)

        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)  # Reshape if it's a single feature

        # Fit the model
        model.fit(X, y)  # Train the model
        joblib.dump(model, 'models/voice_auth_model.pkl')  # Save the trained model
        print("Model trained successfully.")

    except Exception as e:
        print(f"Error during model training: {e}")

def main():
    # Load your data
    X, y = load_your_data()  # Ensure this function is correctly defined in src/train_model.py
    
    # Check if data is empty
    if X.size == 0 or y.size == 0:
        print("No data available for training.")
        return
    
    # Define and train the model
    model = RandomForestClassifier()
    try:
        model.fit(X, y)
        print("Model training completed.")
        
        # Save the trained model for future use
        joblib.dump(model, 'voice_authentication_model.pkl')
        
    except Exception as e:
        print(f"Error during model training: {e}")
        return

    # Record audio
    audio_data = record_audio()  # Ensure you have a record_audio function that returns audio data
    voice_recognition_result = recognize_voice(model, audio_data)
    print("Voice recognition result:", voice_recognition_result)

    # Lip movement detection
    video_path = 0  # Use webcam (or specify a valid video path)
    lip_movement_score = detect_lip_movements(video_path)
    print("Lip movement score:", lip_movement_score)
