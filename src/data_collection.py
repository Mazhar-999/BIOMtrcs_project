import os
import pyaudio
import wave
import cv2
import time

def record_audio(filename, duration=5):
    """Record audio for a given duration and save it to a file."""
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    rate = 44100
    
    audio = pyaudio.PyAudio()
    
    stream = audio.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
    frames = []

    print(f"Recording audio for {filename}...")
    for i in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)
    
    print("Finished recording audio.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

def record_video(filename, duration=5):
    """Record video for a given duration and save it to a file."""
    cap = cv2.VideoCapture(0)  # Use webcam for video input
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))

    print(f"Recording video for {filename}...")
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow('Recording Video', frame)
            if time.time() - start_time > duration:
                break
        else:
            break

    print("Finished recording video.")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    # Set paths for authorized and unauthorized data
    authorized_audio_dir = "data/authorized/voice_samples/"
    authorized_video_dir = "data/authorized/lip_videos/"
    unauthorized_audio_dir = "data/unauthorized/voice_samples/"
    unauthorized_video_dir = "data/unauthorized/lip_videos/"

    # Create directories if they do not exist
    os.makedirs(authorized_audio_dir, exist_ok=True)
    os.makedirs(authorized_video_dir, exist_ok=True)
    os.makedirs(unauthorized_audio_dir, exist_ok=True)
    os.makedirs(unauthorized_video_dir, exist_ok=True)

    # Collect authorized data
    for i in range(1, 3):  # Collect 2 samples for testing
        record_audio(f"{authorized_audio_dir}authorized_sample_{i}.wav")
        record_video(f"{authorized_video_dir}authorized_video_{i}.mp4")

    # Collect unauthorized data
    for i in range(1, 3):  # Collect 2 samples for testing
        record_audio(f"{unauthorized_audio_dir}unauthorized_sample_{i}.wav")
        record_video(f"{unauthorized_video_dir}unauthorized_video_{i}.mp4")

if __name__ == "__main__":
    main()
