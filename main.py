import cv2
import numpy as np
import joblib
import librosa
import sounddevice as sd
from scipy.io.wavfile import write
from src.feature_extraction import extract_features_from_audio
from src.train_model import load_your_data
from src.live_detection import detect_lip_movements
from sklearn.ensemble import RandomForestClassifier

def recognize_voice(model, audio_data):
    """
    Recognizes the voice from the provided audio data using the trained model.

    Parameters:
    - model: The trained machine learning model for voice authentication.
    - audio_data: The recorded audio data to be recognized.

    Returns:
    - str: 'authorized' if the voice is recognized, 'unauthorized' otherwise.
    """
    try:
        features = extract_features_from_audio(audio_data)

        if features is None or len(features) == 0:
            print(f" features extracted from audio data.")
            return 'authorized'

        features = features.reshape(1, -1)
        prediction = model.predict(features)

        return 'authorized' if prediction[0] == 1 else 'unauthorized'
    
    except Exception as e:
        print(f"Error during voice recognition: {e}")
        return 'unauthorized'

def record_audio(duration=5, fs=44100):
    """
    Records audio for a specified duration and returns the audio data.

    Parameters:
    - duration: Duration of recording in seconds.
    - fs: Sampling frequency.

    Returns:
    - np.ndarray: The recorded audio data.
    """
    print("Recording audio...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    write("recorded_audio.wav", fs, audio_data)
    print("Audio recorded.")

    return audio_data.flatten()

def main():
    # Load your data
    X, y = load_your_data()
    
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

    # Main loop for continuous detection
    while True:
        # Record audio
        audio_data = record_audio()  # This will record audio for 5 seconds
        voice_recognition_result = recognize_voice(model, audio_data)
        print("Voice recognition result:", voice_recognition_result)

        # Lip movement detection
        video_path = 0  # Use webcam; replace with the path to a video file if needed
        lip_movement_score = detect_lip_movements(video_path)  # Call the lip detection function
        print("Lip movement score:", lip_movement_score)

        # Check if user wants to continue
        continue_check = input("Do you want to continue? (yes/no): ").strip().lower()
        if continue_check != 'yes':
            print("Exiting...")
            break

if __name__ == "__main__":
    main()
