import os
import cv2
import numpy as np
import pyaudio
import wave
import librosa
import joblib
import speech_recognition as sr
import mediapipe as mp
from .feature_extraction import load_your_data  # Use relative import



# Lip Movement Detection
def detect_lip_movements(video_path=0):
    """
    Detect lip movements from the provided video source.
    If you want to use a webcam, pass 0 as video_path.
    
    Parameters:
    - video_path: The video source (0 for webcam or file path).
    """
    try:
        mp_face_mesh = mp.solutions.face_mesh
        cap = cv2.VideoCapture(video_path)  # Use the specified video source
        
        if not cap.isOpened():
            print("Error: Could not open video.")
            return 0  # Return a default score if video can't be opened

        total_frames = 0
        detected_frames = 0

        with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Error: Could not read frame from video.")
                    break  # Exit the loop if no frame is read
                
                total_frames += 1  # Increment the total frames count
                
                # Convert image to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image_rgb)

                # Draw the lip landmarks if a face is detected
                if results.multi_face_landmarks:
                    detected_frames += 1  # Increment the detected frames count
                    for face_landmarks in results.multi_face_landmarks:
                        for idx in range(61, 68):  # Approximate lip landmarks
                            landmark = face_landmarks.landmark[idx]
                            x = int(landmark.x * image.shape[1])
                            y = int(landmark.y * image.shape[0])
                            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

                # Display the frame with landmarks
                cv2.imshow('Lip Movement Detection', image)
                if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
                    break

        cap.release()
        cv2.destroyAllWindows()
        
        # Calculate and display detection score
        if total_frames > 0:
            detection_score = (detected_frames / total_frames) * 100
            print(f"Lip Movement Detection Score: {detection_score:.2f}%")
            return detection_score
        else:
            print("No frames processed.")
            return 0

    except Exception as e:
        print(f"Error in lip movement detection: {e}")
        return 0  # Return a default score if an error occurs
# Audio Recording Function
def record_audio(filename, duration=5):
    """Record audio for a given duration and save it to a file."""
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    rate = 44100

    audio = pyaudio.PyAudio()

    stream = audio.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
    frames = []

    print(f"Recording {filename}...")
    for i in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print(f"Finished recording {filename}.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Use 'wb' mode to write the frames as binary data
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))


# Feature Extraction from Audio
def extract_features_from_audio(file_path):
    """Extract features from an audio file."""
    signal, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(signal, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Voice Recognition
def recognize_voice():
    """Recognize the voice command using the microphone."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Adjusting for ambient noise, please wait...")
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Adjust for ambient noise
        print("Listening... Please speak the authentication phrase.")
        audio = recognizer.listen(source)

    try:
        spoken_text = recognizer.recognize_google(audio)
        print(f"Recognized Text: {spoken_text}")
        return spoken_text
    except sr.RequestError:
        print("API unavailable or unresponsive")
        return None
    except sr.UnknownValueError:
        print("Unable to recognize speech")
        return None


# User Authentication
def authenticate_user(model):
    """Authenticate user based on voice input."""
    audio_filename = 'data/temp/temp_audio.wav'  # Temporary audio file
    record_audio(audio_filename, duration=5)

    # Extract features and predict
    features = extract_features_from_audio(audio_filename).reshape(1, -1)
    prediction = model.predict(features)

    return prediction[0]  # Return predicted label

# Main Function
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
    audio_data = record_audio()  # This will record audio for 5 seconds
    voice_recognition_result = recognize_voice(model, audio_data)
    print("Voice recognition result:", voice_recognition_result)

    # Lip movement detection
    lip_movement_score = detect_lip_movements(video_path=0)  # Call the integrated lip detection function
    print("Lip movement score:", lip_movement_score)

if __name__ == "__main__":
    main()
