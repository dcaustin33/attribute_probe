from PIL import Image
import requests
import torch

from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn

class CLIP_text_image(nn.Module):

    def __init__(self, args = None):
        super().__init__()
        self.args = args
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.linear1 = nn.Linear(312, 312)
        self.classifier1 = nn.Sequential(nn.Linear(312, 312), nn.ReLU(), nn.BatchNorm1d(312), nn.Linear(312, 312))

    def forward(self, prompts, images):
        text = prompts
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        for i in inputs:
            inputs[i] = inputs[i].cuda()
        inputs['pixel_values'] = images.cuda()
        outputs = self.clip(**inputs)

        binary_pred1 = self.linear1(outputs.logits_per_image)
        classifier_pred1 = self.classifier1(outputs.logits_per_image)


        return (binary_pred1, classifier_pred1), outputs.logits_per_image