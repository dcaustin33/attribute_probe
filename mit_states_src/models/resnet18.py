import torchvision
import torch.nn as nn
import torch
import math

class ResNet18(nn.Module):

    def __init__(self, args = None):
        super().__init__()
        self.model = torchvision.models.resnet18()
        self.model.fc = nn.Identity()
        self.linear = nn.Linear(512, 312)

    def forward(self, images):
        out = self.model(images)
        return self.linear(out)