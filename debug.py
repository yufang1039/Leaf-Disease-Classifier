import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import models as models
from torch.utils.data import DataLoader

from model import model_resnet
from dataset import LeafDataset
from sklearn.metrics import accuracy_score


# Define constant param
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_OF_EPOCH = 20
LEARNING_RATE = 8e-4
BATCH_SIZE = 8


# Get leaf dataset and dataloader for both training and validation dataset
leaf_ds = LeafDataset(csv_file="train_df.csv", imgs_path="train_images/")
ds_len = leaf_ds.__len__()
train_ds, valid_ds = torch.utils.data.random_split(dataset=leaf_ds, 
                    lengths=[math.floor(ds_len * 0.7), ds_len - math.floor(ds_len * 0.7)])
train_loader = DataLoader(train_ds, shuffle=True, batch_size = BATCH_SIZE)
valid_loader = DataLoader(valid_ds, shuffle=True, batch_size = BATCH_SIZE)

TRAIN_SIZE = ds_len
VALID_SIZE = ds_len - math.floor(ds_len * 0.7)
loss_fn = torch.nn.CrossEntropyLoss()

img = leaf_ds.__getitem__(1)[0]
plt.imshow(img)