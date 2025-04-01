import os
import torch
import pandas as pd
import urllib.request
from io import StringIO
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model

# Ensure CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# URL of the dataset (Replace with the actual URL)
# Google Drive File ID (Extract from the shareable link)
file_id = "1rC80Zf_4GUiqUjtI2BnNtmDhSO2YgNgI"
csv_url = f"https://drive.google.com/uc?id={file_id}"

# Load Cardiovascular Dataset directly from the internet
df = pd.read_csv(csv_url, sep=";")

# Convert age from days to years
df["age"] = (df["age"] / 365).astype(int)

# Map categorical values to text
gender_map = {1: "Female", 2: "Male"}
cholesterol_map = {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}
glucose_map = {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}
binary_map = {0: "No", 1: "Yes"}

# Convert each row into a text-based medical record
def row_to_text(row):
    return f"""Patient Record:
- Age: {row['age']} years
- Gender: {gender_map[row['gender']]}
- Height: {row['height']} cm
- Weight: {row['weight']} kg
- Systolic BP: {row['ap_hi']} mmHg
- Diastolic BP: {row['ap_lo']} mmHg
- Cholesterol Level: {cholesterol_map[row['cholesterol']]}
- Glucose Level: {glucose_map[row['gluc']]}
- Smokes: {binary_map[row['smoke']]}
- Alcohol Intake: {binary_map[row['alco']]}
- Physically Active: {binary_map[row['active']]}

Diagnosis: {"Has Cardiovascular Disease" if row['cardio'] == 1 else "No Cardiovascular Disease"}."""

# Apply transformation
df["text"] = df.apply(row_to_text, axis=1)

# Select 500 fixed examples for fine-tuning
df = df.sample(n=min(500, len(df)), random_state=42)  # Select random 500 samples

# Convert to Hugging Face Dataset format
dataset = Dataset.from_dict({"text": df["text"].tolist()})

import os
import torch

# Disable FlashAttention
os.environ["DISABLE_FLASH_ATTN"] = "1"

# Ensure PyTorch does not force FlashAttention
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


# Load BioMedLM-2.7B Model & Tokenizer
model_name = "stanford-crfm/BioMedLM"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# **Fix: Add padding token**
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS as PAD token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # **Fix: Use float16 instead of bfloat16**
    device_map="auto"
)

# Tokenization function
def tokenize_function(examples):
    encoding = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    encoding["labels"] = encoding["input_ids"].clone()
    return encoding

# Apply tokenization
dataset = dataset.map(tokenize_function, batched=True)

# **Fix: Do NOT move dataset to GPU manually**
# Dataset will be handled automatically by Trainer

# Set dataset format
columns_to_keep = ["input_ids", "attention_mask", "labels"]
dataset.set_format(type="torch", columns=columns_to_keep)

# Split into train/test
split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
test_dataset = split["test"]

# **Fix: Update LoRA Config (since BioMedLM lacks `q_proj`, `v_proj`)**
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=["c_proj", "c_attn"]  # Adjusted for BioMedLM
)

# Enable LoRA tuning
model = get_peft_model(model, lora_config)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./BioMedLM-Cardio",
    evaluation_strategy="epoch",
    per_device_train_batch_size=5,  # Adjusted for 80GB A100
    per_device_eval_batch_size=5,   # Adjusted for 80GB A100
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=500,
    fp16=True,  # Enable Mixed Precision
    auto_find_batch_size=True,  # Auto-tune batch size
    gradient_checkpointing=False,  # **Fix: Disabled for stability**
    optim="adamw_torch",
    remove_unused_columns=False,
    seed=42
)

from torch.nn import functional as F
from transformers import Trainer

# Custom Trainer to properly compute loss for causal LM
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Shift logits and labels for causal LM loss computation
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Ensure logits & labels have the same shape
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        loss = F.cross_entropy(
            shift_logits, shift_labels, ignore_index=tokenizer.pad_token_id
        )

        return (loss, outputs) if return_outputs else loss

# Debug CUDA issues
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Clear GPU memory before training
torch.cuda.empty_cache()

# Initialize Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

# Train model
model.train()
trainer.train()

# Save model
model.save_pretrained("./BioMedLM-Cardio")
tokenizer.save_pretrained("./BioMedLM-Cardio")

print("Fine-tuning completed! Model saved at './BioMedLM-Cardio'")


# =========================
# Function to Make Predictions
# =========================
def get_prediction(age, gender, height, weight, ap_hi, ap_lo, cholesterol, glucose, smoke, alco, active):
    input_text = f"""Patient Record:
- Age: {age} years
- Gender: {gender_map[gender]}
- Height: {height} cm
- Weight: {weight} kg
- Systolic BP: {ap_hi} mmHg
- Diastolic BP: {ap_lo} mmHg
- Cholesterol Level: {cholesterol_map[cholesterol]}
- Glucose Level: {glucose_map[glucose]}
- Smokes: {binary_map[smoke]}
- Alcohol Intake: {binary_map[alco]}
- Physically Active: {binary_map[active]}

Diagnosis:"""

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Generate output
    model.eval()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=4)

    # Decode output
    predicted_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract predicted diagnosis
    diagnosis = predicted_text.split("Diagnosis:")[-1].strip()
    
    return diagnosis

# =========================
# **User Input After Training**
# =========================
print("\nEnter Patient Details for Prediction:")
age = int(input("Age (years): "))
gender = int(input("Gender (1=Female, 2=Male): "))
height = int(input("Height (cm): "))
weight = int(input("Weight (kg): "))
ap_hi = int(input("Systolic BP: "))
ap_lo = int(input("Diastolic BP: "))
cholesterol = int(input("Cholesterol Level (1=Normal, 2=Above Normal, 3=Well Above Normal): "))
glucose = int(input("Glucose Level (1=Normal, 2=Above Normal, 3=Well Above Normal): "))
smoke = int(input("Smoker? (0=No, 1=Yes): "))
alco = int(input("Alcohol Intake? (0=No, 1=Yes): "))
active = int(input("Physically Active? (0=No, 1=Yes): "))

# Get Prediction
predicted_diagnosis = get_prediction(age, gender, height, weight, ap_hi, ap_lo, cholesterol, glucose, smoke, alco, active)
print("\nPredicted Diagnosis:", predicted_diagnosis)
