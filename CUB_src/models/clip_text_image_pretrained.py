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

    def forward_image(self, images):
        text =[ "random text"]
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        for i in inputs:
            inputs[i] = inputs[i].cuda()
        inputs['pixel_values'] = images.cuda()
        outputs = self.clip(**inputs)
        return outputs.image_embeds

    def forward_text(self, prompts):
        text = prompts
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        for i in inputs:
            inputs[i] = inputs[i].cuda()
        #give random pixel values
        inputs['pixel_values'] = torch.rand(1, 3, 224, 224).cuda()
        outputs = self.clip(**inputs)
        return outputs.text_embeds


class CLIP_text_image_concat(nn.Module):

    def __init__(self, args = None):
        super().__init__()
        self.args = args
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.linear1 = nn.ModuleList([nn.Linear(1024, 1) for i in range(312)])
        self.linear2 = nn.ModuleList([nn.Linear(768 + 512, 1) for i in range(312)])
        self.classifier1 = nn.ModuleList([nn.Sequential(nn.Linear(1024, 312), nn.ReLU(), nn.BatchNorm1d(312), nn.Linear(312, 1)) for i in range(312)])
        self.classifier2 = nn.ModuleList([nn.Sequential(nn.Linear(768 + 512, 312), nn.ReLU(), nn.BatchNorm1d(312), nn.Linear(312, 1)) for i in range(312)])

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

            lin1 = self.linear1[i](final_embed)
            lin2 = self.linear2[i](final_out)
            class1 = self.classifier1[i](final_embed)
            class2 = self.classifier2[i](final_out)

            inter_class = [lin1, lin2, class1, class2]
            if i == 0:
                classifications = inter_class

            else:
                classifications[0] = torch.cat((classifications[0], inter_class[0]), dim=1)
                classifications[1] = torch.cat((classifications[1], inter_class[1]), dim=1)
                classifications[2] = torch.cat((classifications[2], inter_class[2]), dim=1)
                classifications[3] = torch.cat((classifications[3], inter_class[3]), dim=1)

        return classifications, outputs.logits_per_image



class CLIP_text_image_with_attribute(nn.Module):

    def __init__(self, args = None):
        super().__init__()
        self.args = args
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.linear1 = nn.Linear(1024, 312)
        self.linear2 = nn.Linear(768 + 512, 312)
        self.classifier1 = nn.Sequential(nn.Linear(1024, 312), nn.ReLU(), nn.BatchNorm1d(312), nn.Linear(312, 312))
        self.classifier2 = nn.Sequential(nn.Linear(768 + 512, 312), nn.ReLU(), nn.BatchNorm1d(312), nn.Linear(312, 312))

        self.class_linear1= nn.Linear(1024, 201)
        self.class_classifier1 = nn.Sequential(nn.Linear(1024, 201), nn.ReLU(), nn.BatchNorm1d(201), nn.Linear(201, 201))
        self.class_linear2 = nn.Linear(768 + 512, 201)
        self.class_classifier2 = nn.Sequential(nn.Linear(768 + 512, 201), nn.ReLU(), nn.BatchNorm1d(201), nn.Linear(201, 201))

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

        #now we concatenate the image and text embeddings
        embed = torch.cat((image_embed, text_embed), dim=1)
        out = torch.cat((image_out, text_out), dim=1)

        classifications.append(self.linear1(embed))
        classifications.append(self.linear2(out))
        classifications.append(self.classifier1(embed))
        classifications.append(self.classifier2(out))
        classifications.append(self.class_linear1(embed))
        classifications.append(self.class_linear2(out))
        classifications.append(self.class_classifier1(embed))
        classifications.append(self.class_classifier2(out))


        return classifications, outputs.logits_per_image