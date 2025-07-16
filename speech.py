import whisper
import pyaudio
import numpy as np
import keyboard
from transformers import pipeline

# Load Whisper model (English-only transcription)
model = whisper.load_model("base")

# Load sentiment analysis and emotion detection models
sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# PyAudio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024  # 1024 audio samples per frame.
KEY = "space"

# Initialize PyAudio
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print(f"Press & hold '{KEY}' to record. Press 'Esc' to exit.")

try:
    with open("transcriptions_sentiment.txt", "a", encoding="utf-8") as file:
        while True:
            keyboard.wait(KEY)
            print("Recording... (Release key to stop)")
            frames = []

            while keyboard.is_pressed(KEY):
                frames.append(stream.read(CHUNK, exception_on_overflow=False))

            print("Processing...")

            # Convert recorded frames to a NumPy array & normalize
            audio_np = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0

            # Transcribe speech (English only)
            result = model.transcribe(audio_np, language="en")
            text = result["text"]
            print("Transcribed:", text)

            # Sentiment and emotion analysis
            if text.strip():
                sentiment = sentiment_pipe(text)[0]["label"]
                emotion = emotion_pipe(text)[0]["label"]

                # Store transcribed text along with sentiment and emotion
                file.write(f"{text} | Sentiment: {sentiment} | Emotion: {emotion}\n")
                file.flush()

                print(f"Sentiment: {sentiment}, Emotion: {emotion}")

            if keyboard.is_pressed("esc"):
                break

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()
    print("Goodbye!")
                                                                                                                                                       
