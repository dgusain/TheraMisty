# Install and upgrade necessary packages
# Uncomment the lines below if running in a new environment
# !pip install unsloth
# !pip uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
from datasets import Dataset
from unsloth import is_bfloat16_supported
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
import json
import re

# Suppress warnings and set plot styles
warnings.filterwarnings("ignore")
plt.style.use('ggplot')

# Load the new dataset
data = pd.read_json("dataset_parallel_bigone.json")  # Replace with your actual path

# Inspect the dataset columns
print("Dataset Columns:", data.columns)

# Function to parse convo_script
def parse_convo_script(convo_script):
    """
    Parses the conversation script and returns a list of (Therapist, Student, Background) tuples.
    """
    # Remove Markdown code block delimiters if present
    convo_script = convo_script.strip()
    convo_script = re.sub(r'^```json\s*', '', convo_script)
    convo_script = re.sub(r'```$', '', convo_script)
    
    # Split into lines and remove empty lines
    lines = [line.strip() for line in convo_script.split('\n') if line.strip()]
    
    parsed_convo = []
    i = 0
    while i < len(lines):
        # Extract Therapist line
        therapist_line = lines[i]
        therapist_match = re.match(r'^Therapist:\s*"(.+)"$', therapist_line)
        if therapist_match:
            therapist_message = therapist_match.group(1)
        else:
            print(f"Unexpected format at line {i+1}: {therapist_line}")
            i += 1
            continue  # Skip to next line
        
        # Extract Student line
        if i + 1 < len(lines):
            student_line = lines[i + 1]
            student_match = re.match(r'^Student:\s*"(.+)"$', student_line)
            if student_match:
                student_message = student_match.group(1)
            else:
                print(f"Unexpected format at line {i+2}: {student_line}")
                student_message = ""
        else:
            print(f"Missing Student line after Therapist at line {i+1}")
            student_message = ""
        
        # Extract Background line
        if i + 2 < len(lines):
            background_line = lines[i + 2]
            background_match = re.match(r'^Background:\s*(.+)$', background_line)
            if background_match:
                background_info = background_match.group(1)
            else:
                print(f"Unexpected format at line {i+3}: {background_line}")
                background_info = ""
        else:
            print(f"Missing Background line after Student at line {i+2}")
            background_info = ""
        
        # Append the tuple
        parsed_convo.append((therapist_message, student_message, background_info))
        
        # Move to the next trio
        i += 3
    
    return parsed_convo

# Apply the parsing function
data['parsed_convo'] = data['convo_script'].apply(parse_convo_script)

# Function to create input-output pairs
def create_input_output_pairs(row):
    scene_description = row['therapy_scene_description']
    session_plan = row['session_plan']
    parsed_convo = row['parsed_convo']
    
    input_output_pairs = []
    
    conversation_history = ""
    for idx, (speaker, message) in enumerate(parsed_convo):
        if speaker == "Therapist":
            conversation_history += f"Therapist: {message}\n"
        elif speaker == "Student":
            conversation_history += f"Student: {message}\n"
            # Look ahead for the next therapist message
            if idx + 1 < len(parsed_convo):
                next_speaker, next_message = parsed_convo[idx + 1]
                if next_speaker == "Therapist":
                    input_prompt = f"Therapy Scene Description:\n{scene_description}\n\nSession Plan:\n{session_plan}\n\nConversation History:\n{conversation_history}"
                    response = next_message
                    input_output_pairs.append((input_prompt, response))
                    # Append the therapist message to the conversation history
                    conversation_history += f"Therapist: {response}\n"
    
    return input_output_pairs

# Apply the function to create input-output pairs
all_input_output = data.apply(create_input_output_pairs, axis=1)

# Flatten the list of lists
flat_input_output = [pair for sublist in all_input_output for pair in sublist]

# Convert to DataFrame
fine_tune_data = pd.DataFrame(flat_input_output, columns=['prompt', 'response'])

print(f"Total input-output pairs: {len(fine_tune_data)}")
print(fine_tune_data.head())

# Define maximum sequence lengths
max_prompt_length = 1024  # Adjust based on your model's capability
max_response_length = 512  # Adjust as needed

# Calculate lengths
fine_tune_data['prompt_length'] = fine_tune_data['prompt'].apply(len)
fine_tune_data['response_length'] = fine_tune_data['response'].apply(len)

# Filter out overly long prompts or responses
filtered_data = fine_tune_data[
    (fine_tune_data['prompt_length'] <= max_prompt_length) & 
    (fine_tune_data['response_length'] <= max_response_length)
]

print(f"Total entries after filtering: {len(filtered_data)}")
print(filtered_data[['prompt', 'response']].head())
'''
# Initialize the tokenizer and model
model_name = "unsloth/Llama-3.2-1B-bnb-4bit"  # Replace with your model if different

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_prompt_length,
    load_in_4bit=True,
    dtype=None,
)

# Apply PEFT to the model
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,  # Introduce some dropout to prevent overfitting
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
    use_rslora=True,
    use_gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
    random_state=32,
    loftq_config=None,
)

# Print trainable parameters
print(model.print_trainable_parameters())

# Define the tokenization functions
def tokenize_prompt(examples):
    return tokenizer(examples['prompt'], padding="max_length", truncation=True, max_length=max_prompt_length)

def tokenize_response(examples):
    return tokenizer(examples['response'], padding="max_length", truncation=True, max_length=max_response_length)

# Convert the DataFrame to a Hugging Face Dataset
tokenized_data = Dataset.from_pandas(filtered_data[['prompt', 'response']])

# Tokenize the prompts
tokenized_data = tokenized_data.map(tokenize_prompt, batched=True)

# Tokenize the responses
tokenized_data = tokenized_data.map(tokenize_response, batched=True, remove_columns=["response"])

# Set the labels to be the tokenized responses
def set_labels(batch):
    batch["labels"] = batch["input_ids"].copy()
    return batch

tokenized_data = tokenized_data.map(set_labels, batched=True)

# Remove unnecessary columns
tokenized_data = tokenized_data.remove_columns(["prompt", "prompt_length", "response_length"])

print(tokenized_data)

# Split the dataset (90% train, 10% eval)
split = tokenized_data.train_test_split(test_size=0.1, seed=42)
train_dataset = split['train']
eval_dataset = split['test']

print(f"Training set size: {len(train_dataset)}")
print(f"Evaluation set size: {len(eval_dataset)}")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,  # Increase if needed
    per_device_train_batch_size=4,  # Adjust based on GPU memory
    per_device_eval_batch_size=4,
    logging_dir='./logs',
    logging_steps=10,
    fp16=True,  # Use mixed precision if supported
    gradient_accumulation_steps=4,  # To simulate a larger batch size
    evaluation_strategy="epoch",
    save_strategy="epoch",
    report_to="none",  # Disable reporting to external systems like WandB
)

# Initialize the SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="prompt",
)

# Start training
trainer.train()

# Define the output directory
output_dir = "./fine-tuned-therapy-model"

# Save the model and tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer saved to {output_dir}")
'''