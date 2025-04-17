from transformers import pipeline
import soundfile as sf
import librosa
import numpy as np

# Load the fine-tuned model
emotion_classifier = pipeline(
    "audio-classification", 
    model="./final_model",
    feature_extractor="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)

# Function to predict emotion from audio file
def predict_emotion(audio_path):
    # Load audio
    audio, sampling_rate = librosa.load(audio_path, sr=16000)
    
    # Make prediction
    result = emotion_classifier(audio)
    
    # Print results
    print(f"Audio: {audio_path}")
    for pred in result:
        print(f"Emotion: {pred['label']}, Score: {pred['score']:.4f}")
    
    return result

# Test with a few samples
test_file = "/home/polo/SER/test-ser/ravdess/Actor_03/03-01-05-02-02-02-03.wav"  # Replace with your test file
predict_emotion(test_file)