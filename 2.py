import torch
from datasets import load_from_disk, Audio
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer
)
import evaluate
import numpy as np

# Load dataset
ravdess_dataset = load_from_disk("ravdess_dataset")

# Get label information
label_list = ravdess_dataset["train"].unique("emotion")
label_list.sort()  # Sort labels for consistency
num_labels = len(label_list)
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

print(f"Labels: {label_list}")
print(f"Number of labels: {num_labels}")

# Load model and feature extractor
model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label
)

# Prepare dataset with feature extraction
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["path"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate,
        max_length=16000*5,  # 5 seconds max
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    inputs["labels"] = [label2id[label] for label in examples["emotion"]]
    return inputs

# Apply preprocessing
encoded_dataset = ravdess_dataset.map(
    preprocess_function,
    remove_columns=["path"],
    batch_size=8,
    batched=True
)

# Define metrics
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # changed from evaluation_strategy to eval_strategy
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,  # Set to True if you want to push to Hugging Face Hub
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Save the model
trainer.save_model("./final_model")