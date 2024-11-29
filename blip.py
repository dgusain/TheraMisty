# pip install accelerate
import torch
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import time
import io

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



if __name__ == "__main__":
    device = torch.device("cuda:0")
    start_time = time.time()
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)
    load_dur = time.time() - start_time
    print(f"Model loaded in {load_dur} seconds")



    #img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
    #raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    img = "rabbit.jpg"
    raw_image = Image.open(img).convert("RGB")
    #save_path = "user_pic.jpg"
    #start_time = time.time()
    #download_image(url, save_path)
    #image = Image.open(save_path).convert("RGB")
    #img_time = time.time() - start_time
    start_time = time.time()
    question = "How many rabbits are in this picture?"
    inputs = processor(raw_image, question, return_tensors="pt",truncation=True,max_length=4096).to(model.device)
    out = model.generate(**inputs,max_new_tokens=100)
    print(out)
    inf_time = time.time() - start_time
    resp = processor.decode(out[0], skip_special_tokens=True)
    print(resp)
    print(f"Inference time: {inf_time} seconds")


'''
img_url = '' 
"http://67.20.193.16/api/cameras/rgb?base64=false&fileName=user_pic&displayOnScreen=false&overwriteExisting=false"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
'''



