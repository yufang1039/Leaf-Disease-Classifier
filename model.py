import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as models
from efficientnet_pytorch import EfficientNet


def model_resnet():
    # Get the resenet  from model API
    model = models.resnet50(pretrained=None)

    model.fc = nn.Linear(in_features=2048, out_features=6)
    return model  

def model_eff():
    model = EfficientNet.from_pretrained('efficientnet-b4')
    layers = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(1000, 6),
        nn.Softmax(dim=1)
    )
    return nn.Sequential(model, layers)

model = model_eff()
print(model)