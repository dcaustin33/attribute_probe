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


class CLIP_text_image_concat(nn.Module):

    def __init__(self, adjectives: int, nouns: int, concat: int, args = None):
        super().__init__()
        self.args = args
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.linear_adjectives = nn.ModuleList([nn.Linear(768 + 512, 1) for i in range(adjectives)])
        self.linear_nouns = nn.ModuleList([nn.Linear(768 + 512, 1) for i in range(nouns)])
        self.linear_concat = nn.ModuleList([nn.Linear(768 + 512, 1) for i in range(concat)])

        self.classifier_adjectives =nn.ModuleList([nn.Sequential(nn.Linear(768 + 512, 312), nn.ReLU(), nn.BatchNorm1d(312), nn.Linear(312, 1)) for i in range(adjectives)])
        self.classifier_nouns = nn.ModuleList([nn.Sequential(nn.Linear(768 + 512, 312), nn.ReLU(), nn.BatchNorm1d(312), nn.Linear(312, 1)) for i in range(nouns)])
        self.classifier_concat = nn.ModuleList([nn.Sequential(nn.Linear(768 + 512, 312), nn.ReLU(), nn.BatchNorm1d(312), nn.Linear(312, 1)) for i in range(concat)])

    def forward(self, prompts, images):
        text = prompts
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        for i in inputs:
            inputs[i] = inputs[i].cuda()
        inputs['pixel_values'] = images.cuda()
        outputs = self.clip(**inputs)

        image_embed = outputs.image_embeds
        image_out = outputs.vision_model_output['pooler_output']
        text_embed = outputs.text_embeds
        text_out = outputs.text_model_output['pooler_output']

        classifications = []

        for i in range(text_embed.size(0)):
            new_text_embed = text_embed[i].unsqueeze(0).repeat(image_embed.size(0), 1)
            new_text_out = text_out[i].unsqueeze(0).repeat(image_out.size(0), 1)
            final_embed = torch.cat((image_embed, new_text_embed), dim=1)
            final_out = torch.cat((image_out, new_text_out), dim=1)

            adjective_linear = self.linear_adjectives[i](final_out)
            noun_linear = self.linear_nouns[i](final_out)
            concat_linear = self.linear_concat[i](final_out)

            adjective_classifier = self.classifier_adjectives[i](final_out)
            noun_classifier = self.classifier_nouns[i](final_out)
            concat_classifier = self.classifier_concat[i](final_out)


            inter_class = [adjective_linear, noun_linear, concat_linear, adjective_classifier, noun_classifier, concat_classifier]
            if i == 0:
                classifications = inter_class

            else:
                classifications[0] = torch.cat((classifications[0], inter_class[0]), dim=1)
                classifications[1] = torch.cat((classifications[1], inter_class[1]), dim=1)
                classifications[2] = torch.cat((classifications[2], inter_class[2]), dim=1)
                classifications[3] = torch.cat((classifications[3], inter_class[3]), dim=1)
                classifications[4] = torch.cat((classifications[4], inter_class[4]), dim=1)
                classifications[5] = torch.cat((classifications[5], inter_class[5]), dim=1)

        return classifications, outputs.logits_per_image