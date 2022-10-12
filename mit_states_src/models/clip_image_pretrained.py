from PIL import Image
import requests
import torch

from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn

class CLIP_image(nn.Module):

    def __init__(self, adjectives: int, nouns: int, concat: int, args = None):
        super().__init__()
        self.args = args
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.linear_adjectives = nn.Linear(768, adjectives)
        self.linear_nouns = nn.Linear(768, nouns)
        self.linear_concat = nn.Linear(768, concat)

        self.classifier_adjectives =nn.Sequential(nn.Linear(768, 312), nn.ReLU(), nn.BatchNorm1d(312), nn.Linear(312, adjectives))
        self.classifier_nouns = nn.Sequential(nn.Linear(768, 312), nn.ReLU(), nn.BatchNorm1d(312), nn.Linear(312, nouns))
        self.classifier_concat = nn.Sequential(nn.Linear(768, 312), nn.ReLU(), nn.BatchNorm1d(312), nn.Linear(312, concat))

    def forward(self, images):
        text = ['']
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        for i in inputs:
            inputs[i] = inputs[i].cuda()
        inputs['pixel_values'] = images.cuda()
        outputs = self.clip(**inputs)

        final_out = outputs.vision_model_output['pooler_output']

        adjective_linear = self.linear_adjectives(final_out)
        noun_linear = self.linear_nouns(final_out)
        concat_linear = self.linear_concat(final_out)

        adjective_classifier = self.classifier_adjectives(final_out)
        noun_classifier = self.classifier_nouns(final_out)
        concat_classifier = self.classifier_concat(final_out)

        classifications = [adjective_linear, noun_linear, concat_linear, adjective_classifier, noun_classifier, concat_classifier]


        return classifications, outputs.logits_per_image