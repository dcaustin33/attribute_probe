import torch
from torchvision import models

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
    def __init__(self, device = 'cuda'):
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
        self.linear1 = torch.nn.Linear(2048, 312)
        self.classifier1 = torch.nn.Sequential(torch.nn.Linear(2048, 312), torch.nn.ReLU(), torch.nn.BatchNorm1d(312), torch.nn.Linear(312, 312))
    
    def forward(self, x):
        x = self.model.encoder(x)
        x = torch.flatten(x, 1)
        classifier_out = self.classifier1(x)
        linear_out = self.linear1(x)
        return (linear_out, classifier_out)