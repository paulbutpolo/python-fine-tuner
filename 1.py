import os
import pandas as pd
import numpy as np
from datasets import Dataset, Audio

def extract_emotion_from_filename(filename):
    # Parse the filename to extract emotion
    parts = filename.split('.')[0].split('-')
    emotion_code = int(parts[2])
    
    # Map emotion code to label
    emotion_map = {
        1: "neutral",
        2: "calm",
        3: "happy", 
        4: "sad",
        5: "angry",
        6: "fearful",
        7: "disgust",
        8: "surprised"
    }
    
    return emotion_map[emotion_code]

def load_ravdess_data(data_dir):
    file_paths = []
    labels = []
    
    # Walk through all actor directories
    for actor_dir in sorted(os.listdir(data_dir)):
        if actor_dir.startswith("Actor_"):
            actor_path = os.path.join(data_dir, actor_dir)
            for filename in os.listdir(actor_path):
                if filename.endswith(".wav"):
                    # Only include audio-only files (03)
                    if filename.startswith("03-"):
                        file_path = os.path.join(actor_path, filename)
                        emotion = extract_emotion_from_filename(filename)
                        
                        file_paths.append(file_path)
                        labels.append(emotion)
    
    # Create dataframe
    data = {
        "path": file_paths,
        "emotion": labels
    }
    
    df = pd.DataFrame(data)
    return df

# Load the data
ravdess_df = load_ravdess_data("/home/polo/SER/test-ser/ravdess")

# Convert to HuggingFace dataset
ravdess_dataset = Dataset.from_pandas(ravdess_df)

# Add audio feature
ravdess_dataset = ravdess_dataset.cast_column("path", Audio())

# Split data into train and validation sets (80/20 split)
ravdess_dataset = ravdess_dataset.train_test_split(test_size=0.2, seed=42)

# Save the dataset
ravdess_dataset.save_to_disk("ravdess_dataset")