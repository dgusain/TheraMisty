# pip install accelerate
import torch
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess
import time

img = "rabbit.jpg"
raw_image = Image.open(img).convert("RGB")
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
start_time = time.time()
# Other available models:
# 
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device
)
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_opt", model_type="pretrain_opt6.7b", is_eval=True, device=device
# )
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=device
# )
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_opt", model_type="caption_coco_opt6.7b", is_eval=True, device=device
# )
#
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device
# )
#
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_t5", model_type="caption_coco_flant5xl", is_eval=True, device=device
# )
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device
# )
load_time = time.time() - start_time
start_time = time.time()
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
question = "What does this image represent?"
response = model.generate({"image": image, "prompt": f"Question:{question} Answer:"})
inf_time = time.time()-start_time
print(response)
print(f"Model loaded in {load_time} seconds")
print(f"Inference time: {inf_time} seconds")

'''
device = torch.device("cuda:0")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(device)

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
#img = "rabbit.jpg"
#raw_image = Image.open(img).convert("RGB")

question = "how many dogs are in the picture?"
inputs = processor(raw_image, question, return_tensors="pt",truncation=True, max_length = 512).to("cuda")
print(inputs)
out = model.generate(**inputs, max_new_tokens=50)
print(out)
print(processor.decode(out[0], skip_special_tokens=True).strip())
'''