from PIL import Image
import requests
import torch

from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn


class CLIP_image(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.linear = nn.Linear(512, 312)

    def forward(self, images):
        text = ['']
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        inputs['pixel_values'] = images
        outputs = self.clip(**inputs)
        binary_pred = self.linear(outputs.image_embeds)
        return binary_pred