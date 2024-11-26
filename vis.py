import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor  
import os
import time
from typing import List, Optional, Dict
import io
import requests

def download_image(url: str, save_path: str, timeout: int = 10):
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()  
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        image.save(save_path)
        print(f"Image successfully downloaded and saved as {save_path}")
        

    except requests.exceptions.Timeout:
        print(f"Error: The request timed out after {timeout} seconds.")
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")  # e.g., 404 Not Found
    except requests.exceptions.RequestException as req_err:
        print(f"Error during request: {req_err}")  # Other request-related errors
    except IOError as io_err:
        print(f"IO error occurred while saving the image: {io_err}")  # File saving errors
    except Exception as e:
        print(f"An unexpected error occurred: {e}")  # Any other exceptions


model_id = "meta-llama/Llama-3.2-11B-Vision"
MAIN_CACHE_DIR: str = "/home/dgusain/misty/Huggingface/"  
MODEL_NAME: str = "meta-llama/Llama-3.2-11B-Vision-Instruct"
LOCAL_MODEL_DIR: str = os.path.join(MAIN_CACHE_DIR, MODEL_NAME)  
HUGGINGFACE_TOKEN: Optional[str] = os.getenv("HUGGINGFACE_TOKEN")
device = torch.device("cuda:0")

start_time = time.time()

model = MllamaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        cache_dir=MAIN_CACHE_DIR,
        use_auth_token=HUGGINGFACE_TOKEN,  
        torch_dtype=torch.float16,  
).half().to(device)


# Load the processor
processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        cache_dir=MAIN_CACHE_DIR,
        use_auth_token=HUGGINGFACE_TOKEN,  
)
load_time = time.time() - start_time
print(f"Model loaded in {load_time} seconds")
'''
# Load the local image
image_path = 'rabbit.jpg'  # Ensure this path is correct relative to your script
try:
    image = Image.open(image_path).convert("RGB")  # Convert to RGB to ensure consistency
except FileNotFoundError:
    print(f"Error: The file {image_path} was not found.")
    exit(1)
except Exception as e:
    print(f"An error occurred while opening the image: {e}")
    exit(1)
'''

if __name__ == "__main__":
    url = "http://67.20.193.16/api/cameras/rgb?base64=false&fileName=user_pic&displayOnScreen=false&overwriteExisting=false"
    save_path = "user_pic.jpg"
    start_time = time.time()
    download_image(url, save_path)
    image = Image.open(save_path).convert("RGB")
    img_time = time.time() - start_time

    # Define the prompt
    prompt = "<|image|><|begin_of_text|>What does this image represent?"
    start_time = time.time()
    # Process the inputs
    inputs = processor(
        image,
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096  # Ensure this aligns with your Config.MAX_INPUT_LENGTH
    ).to(model.device)

    # Generate the response
    output = model.generate(**inputs, max_new_tokens=30)

    # Decode and print the response
    decoded_output = processor.decode(output[0], skip_special_tokens=True)
    print(decoded_output)
    inf_time = time.time() - start_time
    print(f"Image captured in {img_time} seconds")
    print(f"Model inference in {inf_time} seconds")