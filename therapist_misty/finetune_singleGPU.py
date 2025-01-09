# Import necessary libraries
import os
import re
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset
from trl import SFTTrainer
from unsloth import FastLanguageModel  # Ensure unsloth is installed correctly
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Define the parse_convo_script function
def parse_convo_script(convo_script):
    convo_script = convo_script.strip()
    convo_script = re.sub(r'^```json\s*', '', convo_script)
    convo_script = re.sub(r'```$', '', convo_script)
    lines = [line.strip() for line in convo_script.split('\n') if line.strip()]
    
    parsed_convo = []
    i = 0
    while i < len(lines):
        therapist_line = lines[i]
        therapist_match = re.match(r'^Therapist:\s*"(.+)"$', therapist_line)
        if therapist_match:
            therapist_message = therapist_match.group(1)
        else:
            print(f"Unexpected format at line {i+1}: {therapist_line}")
            i += 1
            continue  

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

# Step 1: Load the Dataset
def load_dataset(file_path):
    try:
        data = pd.read_json(file_path)
        print(f"Dataset loaded successfully with {len(data)} entries.")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure

# Step 2: Parse the convo_script
def process_conversations(data):
    data['parsed_convo'] = data['convo_script'].apply(
        lambda x: parse_convo_script(x) if isinstance(x, str) else []
    )

    filtered_data = data[data['parsed_convo'].apply(len) > 0].copy()
    print(f"Total entries after filtering empty conversations: {len(filtered_data)}")
    return filtered_data

# Step 3: Create Input-Output Pairs
def create_input_output_pairs(row):
    scene_description = row['therapy_scene_description']
    session_plan = row['session_plan']
    parsed_convo = row['parsed_convo']
    
    input_output_pairs = []
    
    conversation_history = ""
    for idx, (therapist_msg, student_msg, background_info) in enumerate(parsed_convo):
        # Append Therapist and Student messages to the conversation history
        conversation_history += f"Therapist: {therapist_msg}\n"
        conversation_history += f"Student: {student_msg}\n"
        
        # Prepare the prompt up to the Student's response
        input_prompt = (
            f"Therapy Scene Description:\n{scene_description}\n\n"
            f"Session Plan:\n{session_plan}\n\n"
            f"Conversation History:\n{conversation_history}"
            f"Therapist:"
        )
        
        # Check if there's a next Therapist response
        if idx + 1 < len(parsed_convo):
            next_therapist_msg, _, next_background_info = parsed_convo[idx + 1]
            # Format the response to include both Therapist message and Background info
            output_response = f"Therapist: \"{next_therapist_msg}\"\nBackground: {next_background_info}"
            input_output_pairs.append((input_prompt, output_response))
        
    return input_output_pairs

def generate_fine_tune_data(filtered_data):
    all_input_output = filtered_data.apply(create_input_output_pairs, axis=1)
    
    # Flatten the list of lists
    flat_input_output = [pair for sublist in all_input_output for pair in sublist]
    
    # Create DataFrame
    fine_tune_data = pd.DataFrame(flat_input_output, columns=['prompt', 'response'])
    
    print(f"Total input-output pairs: {len(fine_tune_data)}")
    print(fine_tune_data.head())
    
    return fine_tune_data

# Step 4.1: Calculate Average Sequence Length per Pair
def calculate_avg_sequence_length(fine_tune_data, tokenizer):
    def get_token_length(text):
        return len(tokenizer.encode(text, truncation=False, add_special_tokens=True))
    
    fine_tune_data['prompt_tokens'] = fine_tune_data['prompt'].apply(get_token_length)
    fine_tune_data['response_tokens'] = fine_tune_data['response'].apply(get_token_length)
    fine_tune_data['total_tokens'] = fine_tune_data['prompt_tokens'] + fine_tune_data['response_tokens']
    
    avg_prompt_length = fine_tune_data['prompt_tokens'].mean()
    avg_response_length = fine_tune_data['response_tokens'].mean()
    avg_total_length = fine_tune_data['total_tokens'].mean()
    
    print(f"Average Prompt Token Length: {avg_prompt_length:.2f}")
    print(f"Average Response Token Length: {avg_response_length:.2f}")
    print(f"Average Total Token Length: {avg_total_length:.2f}")
    
    return avg_prompt_length, avg_response_length, avg_total_length


# Step 4: Filter Based on Sequence Lengths
def filter_by_length(fine_tune_data, max_prompt_length=6144, max_response_length=6144):
    fine_tune_data['prompt_length'] = fine_tune_data['prompt'].apply(len)
    fine_tune_data['response_length'] = fine_tune_data['response'].apply(len)
    
    filtered_fine_tune_data = fine_tune_data[
        (fine_tune_data['prompt_length'] <= max_prompt_length) & 
        (fine_tune_data['response_length'] <= max_response_length)
    ].copy()
    
    print(f"Total entries after filtering by length: {len(filtered_fine_tune_data)}")
    return filtered_fine_tune_data

# Step 5: Tokenize the Data
def tokenize_data(filtered_fine_tune_data, tokenizer, max_prompt_length=1024, max_response_length=512):
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(filtered_fine_tune_data[['prompt', 'response']])
    
    # Define tokenization functions
    def tokenize_prompt(examples):
        return tokenizer(
            examples['prompt'],
            padding="max_length",
            truncation=True,
            max_length=max_prompt_length
        )
    
    def tokenize_response(examples):
        return tokenizer(
            examples['response'],
            padding="max_length",
            truncation=True,
            max_length=max_response_length
        )
    
    # Tokenize prompts
    dataset = dataset.map(tokenize_prompt, batched=True)
    
    # Tokenize responses
    dataset = dataset.map(tokenize_response, batched=True, remove_columns=["response"])
    
    # Set labels to be the tokenized responses
    def set_labels(batch):
        batch["labels"] = batch["input_ids"].copy()
        return batch
    
    dataset = dataset.map(set_labels, batched=True)
    
    # Remove unnecessary columns
    #dataset = dataset.remove_columns(["prompt", "prompt_length", "response_length"])
    
    print("Tokenization complete. Sample tokenized data:")
    print(dataset[0])
    
    return dataset

# Step 6: Initialize Tokenizer and Model with PEFT
def initialize_model(model_name, max_prompt_length=1024):
    # Initialize the tokenizer and model using Unsloth
    #tokenizer = AutoTokenizer.from_pretrained(model_name)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=64000,
        load_in_4bit=True,  # Adjust based on your hardware capabilities
        dtype=None,
    )

    #model = model.to(dtype=torch.bfloat16) # explicit setting. 
    
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
    
    # Print trainable parameters to verify PEFT setup
    print("Trainable Parameters:")
    print(model.print_trainable_parameters())
    
    return model, tokenizer

# Step 7: Split the Dataset
def split_dataset(tokenized_dataset):
    split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split['train']
    eval_dataset = split['test']
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Evaluation set size: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset

# Step 8: Define Training Arguments
def define_training_args():
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,  # Adjust based on dataset size and desired performance
        per_device_train_batch_size=1,  # Adjust based on GPU memory
        per_device_eval_batch_size=1,
        logging_dir='./logs',
        logging_steps=10,
        fp16=True,  # Use mixed precision if supported
        gradient_accumulation_steps=8,  # To simulate a larger batch size
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=5,  # Keep only the last 5 checkpoints
        report_to="none",  # Disable reporting to external systems like WandB
    )
    
    return training_args

# Step 9: Initialize the Trainer
def initialize_trainer(model, tokenizer, training_args, train_dataset, eval_dataset):
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="prompt",
    )
    
    return trainer

# Step 10: Train the Model
def train_model(trainer):
    print("Starting training...")
    trainer.train()
    print("Training complete.")
    return trainer

# Step 11: Save the Fine-Tuned Model
def save_model_and_tokenizer(trainer, output_dir="./fine-tuned-therapy-model"):
    trainer.save_model(output_dir)
    trainer.tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")


# Main Execution Flow
def main():
    dataset_path = "dataset_parallel_bigone.json" 
    data = load_dataset(dataset_path)
    if data.empty:
        print("Dataset is empty or failed to load. Exiting.")
        return

    filtered_data = process_conversations(data)
    if filtered_data.empty:
        print("No valid conversations found after parsing. Exiting.")
        return

    fine_tune_data = generate_fine_tune_data(filtered_data)
    if fine_tune_data.empty:
        print("No input-output pairs generated. Exiting.")
        return
    
    #model_name = "unsloth/Llama-3.2-1B-bnb-4bit"
    #tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1b-instruct")
    #p, r, t = calculate_avg_sequence_length(fine_tune_data, tokenizer)
    #print(f"Average: prompt length: {p}, response length: {r}, total length: {t}")

    max_prompt_length = 64000 #16384  # Adjust based on your model's capability
    max_response_length = 1024  # Adjust as needed
    filtered_fine_tune_data = filter_by_length(fine_tune_data, max_prompt_length, max_response_length)
    if filtered_fine_tune_data.empty:
        print("No data left after filtering by length. Exiting.")
        return
    
    # Step 5: Initialize Tokenizer and Model with PEFT
    #model_name = "unsloth/Llama-3.2-1B-bnb-4bit"  # Replace with your desired model
    model_name = "meta-llama/Llama-3.2-1b-instruct"
    model, tokenizer = initialize_model(model_name, max_prompt_length)
    
    # Step 6: Tokenize the Data
    tokenized_dataset = tokenize_data(filtered_fine_tune_data, tokenizer, max_prompt_length, max_response_length)
    
    # Step 7: Split the Dataset
    train_dataset, eval_dataset = split_dataset(tokenized_dataset)
    
    # Step 8: Define Training Arguments
    training_args = define_training_args()
    
    # Step 9: Initialize the Trainer
    trainer = initialize_trainer(model, tokenizer, training_args, train_dataset, eval_dataset)
    
    # Step 10: Train the Model
    trainer = train_model(trainer)
    
    # Step 11: Save the Fine-Tuned Model
    save_model_and_tokenizer(trainer, output_dir="./fine-tuned-therapy-model")
    
if __name__ == "__main__":
    main()
