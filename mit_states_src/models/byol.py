import torch
from torchvision import models
import torch.nn as nn

class ResNet(torch.nn.Module):
    def __init__(self, net_name, weights = None, use_fc=False):
        super().__init__()
        base_model = models.__dict__[net_name](weights = weights)
        self.encoder = torch.nn.Sequential(*list(base_model.children())[:-1])

        self.use_fc = use_fc
        if self.use_fc:
            self.fc = torch.nn.Linear(2048, 312)
            

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        if self.use_fc:
            x = self.fc(x)
        return x

'''device = torch.device('cpu')
model = ResNet('resnet50', pretrained=False, use_fc=True).to(device)

# load encoder
model_path = 'resnet50_byol_imagenet2012.pth.tar'
checkpoint = torch.load(model_path, map_location=device)['online_backbone']
state_dict = {}
length = len(model.encoder.state_dict())
for name, param in zip(model.encoder.state_dict(), list(checkpoint.values())[:length]):
    state_dict[name] = param
model.encoder.load_state_dict(state_dict, strict=True)
model.eval()

example = torch.ones(1, 3, 224, 224)

# convert to torch.jit.ScriptModule via tracing
traced_script_module = torch.jit.trace(model, example)
for p in traced_script_module.parameters():
    p.requires_grad = False

print(traced_script_module)
traced_script_module.save('resnet50_byol_imagenet2012.pt')

assert (model(example) == traced_script_module(example)).all()'''
import os

class BYOL(torch.nn.Module):
    def __init__(self, adjectives: int, nouns: int, concat: int, device = 'cuda'):
        super().__init__()
        model = ResNet('resnet50', weights = None, use_fc=True)

        model_path = '../../models/resnet50_byol_imagenet2012.pth.tar'
        checkpoint = torch.load(model_path, map_location=device)['online_backbone']
        state_dict = {}
        length = len(model.encoder.state_dict())
        for name, param in zip(model.encoder.state_dict(), list(checkpoint.values())[:length]):
            state_dict[name] = param
        model.encoder.load_state_dict(state_dict, strict=True)

        example = torch.ones(1, 3, 224, 224)

        # convert to torch.jit.ScriptModule via tracing
        traced_script_module = torch.jit.trace(model, example)
        for p in traced_script_module.parameters():
            p.requires_grad = False

        self.model = model

        self.linear_adjectives = nn.Linear(2048, adjectives)
        self.linear_nouns = nn.Linear(2048, nouns)
        self.linear_concat = nn.Linear(2048, concat)

        self.classifier_adjectives =nn.Sequential(nn.Linear(2048, 312), nn.ReLU(), nn.BatchNorm1d(312), nn.Linear(312, adjectives))
        self.classifier_nouns = nn.Sequential(nn.Linear(2048, 312), nn.ReLU(), nn.BatchNorm1d(312), nn.Linear(312, nouns))
        self.classifier_concat = nn.Sequential(nn.Linear(2048, 312), nn.ReLU(), nn.BatchNorm1d(312), nn.Linear(312, concat))

    
    def forward(self, x):
        x = self.model.encoder(x)
        final_out = torch.flatten(x, 1)
        adjective_linear = self.linear_adjectives(final_out)
        noun_linear = self.linear_nouns(final_out)
        concat_linear = self.linear_concat(final_out)

        adjective_classifier = self.classifier_adjectives(final_out)
        noun_classifier = self.classifier_nouns(final_out)
        concat_classifier = self.classifier_concat(final_out)

        classifications = [adjective_linear, noun_linear, concat_linear, adjective_classifier, noun_classifier, concat_classifier]
        return (classifications, final_out)