from PIL import Image
import requests
import torch

from transformers import ViltProcessor, ViltModel, ViltForMaskedLM
import torch.nn as nn

class ViLT(nn.Module):

    def __init__(self, args = None):
        super().__init__()
        self.args = args
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        self.model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm-itm")
        self.linear1 = nn.Linear(768, 312)
        self.classifier1 = nn.Sequential(nn.Linear(768, 312), nn.ReLU(), nn.BatchNorm1d(312), nn.Linear(312, 312))

        self.class_linear1= nn.Linear(768, 201)
        self.class_classifier1 = nn.Sequential(nn.Linear(768, 201), nn.ReLU(), nn.BatchNorm1d(201), nn.Linear(201, 201))

    def forward(self, prompts, images):
        images = [i.cpu() for i in images]
        inputs = self.processor(images, prompts, return_tensors="pt", padding = True)
        for i in inputs:
            inputs[i] = inputs[i].cuda()
        outputs = self.model(**inputs)
        out = outputs.pooler_output

        classifications = []

        classifications.append(self.linear1(out))
        classifications.append(self.classifier1(out))
        classifications.append(self.class_linear1(out))
        classifications.append(self.class_classifier1(out))


        return classifications


class ViLT_MLM(nn.Module):

    def __init__(self, args = None):
        super().__init__()
        self.args = args
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        self.model = ViltForMaskedLM.from_pretrained("dandelin/vilt-b32-mlm-itm")

    def forward(self, prompts, images):
        images = [i.cpu() for i in images]
        inputs = self.processor(images, prompts, return_tensors="pt", padding = True)
        for i in inputs:
            inputs[i] = inputs[i].cuda()

        return self.model(**inputs)
