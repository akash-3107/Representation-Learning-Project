import os
import cv2
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import json
import torch

def generate_image_caption(image_path, processor):    
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt").to(device)
    caption = model.generate(**inputs)
    caption_text = processor.decode(caption[0], skip_special_tokens=True)
    return caption_text

print('Started.....')
base_dir = "/home/hpc/rlvl/rlvl129v/ControlNet/training/ade20k/ADE20K_2021_17_01/images/ADE/training"
image_files =[]
mask_files = []
data_pairs = []
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model.to(device)

for root, dirs, files in os.walk(base_dir):
    #image_files = [f for f in files if f.endswith('.jpg')]
    for f in files:
        if f.endswith(".jpg"):
            image_path = os.path.join(root,f)
            mask_path = os.path.join(root, f.split('.')[0] + '_seg.png')
            #image_files.append(image_path)
            #mask_files.append(mask_path)
            caption = generate_image_caption(image_path, processor)
            print(caption)
            data_pairs.append((image_path, mask_path, caption))




formatted_data = [
    {"source": src, "target": tgt, "prompt": prompt}
    for src, tgt, prompt in data_pairs
]


output_json = "ade20k_caption_pairs_2.json"
with open(output_json, "w") as f:
    json.dump(formatted_data, f)            


'''
image_dir = "/home/hpc/rlvl/rlvl129v/ControlNet/training/ade20k/ADE20K_2021_17_01/images/ADE/training/transportation/airport_terminal"
data_pairs = []

jpg_files = [f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")]

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

for j in jpg_files:
    image_path = os.path.join(image_dir, j)
    mask_path = os.path.join(image_dir, j.split('.')[0] + '_seg.png')
    #caption = generate_image_caption(image_path, processor)
    #data_pairs.append((image_path, mask_path, caption))
'''

'''
formatted_data = [
    {"source": src, "target": tgt, "prompt": prompt}
    for src, tgt, prompt in data_pairs
]


output_json = "ade20k_caption_pairs.json"
with open(output_json, "w") as f:
    json.dump(formatted_data, f, indent=4)
'''