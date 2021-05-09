import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import models as models

def model_resnet():
    # Get the resenet  from model API
    model = models.resnet50(pretrained=None)

    model.fc = nn.Linear(in_features=2048, out_features=6)
    return model  