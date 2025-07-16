# Speech-to-Text-and-Emotion-Tracking-System
Convert spoken speech into text using OpenAI's Whisper and analyzes emotional sentiment using DistilBERT and emotion-classifying using DistilRoBERTa model. Implements a simple interface with audio recording and live output.
## Working
The user initiates recording through a key press.
The system captures real-time audio using PyAudio.
The audio is processed and passed to the Whisper model.
Whisper transcribes the speech into text.
The text is sent to models DistilBERT and DistilRoBERTa
DistilBERT for sentiment analysis (positive/negative)


