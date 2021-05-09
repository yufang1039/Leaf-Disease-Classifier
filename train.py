import math
import numpy as np
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
leaf_ds = LeafDataset(csv_file="train_df.csv", imgs_path="train_images/", 
                transform=torchvision.transforms.CenterCrop([1700, 1700]))
ds_len = leaf_ds.__len__()
train_ds, valid_ds = torch.utils.data.random_split(dataset=leaf_ds, 
                    lengths=[math.floor(ds_len * 0.7), ds_len - math.floor(ds_len * 0.7)])
train_loader = DataLoader(train_ds, shuffle=True, batch_size = BATCH_SIZE)
valid_loader = DataLoader(valid_ds, shuffle=True, batch_size = BATCH_SIZE)

TRAIN_SIZE = math.floor(ds_len * 0.7)
VALID_SIZE = ds_len - math.floor(ds_len * 0.7)
loss_fn = torch.nn.BCEWithLogitsLoss()

## Train funciton that train the model for one epoche
def train_fn(net, loader):
    
    running_loss = 0
    preds_for_acc = []
    labels_for_acc = []
    
    for _, (images, labels) in enumerate(loader):
        
        images, labels = images.to(device), labels.to(device)

        ## Set mode to training
        net.train()

        ## Set gradient to zero
        optimizer.zero_grad()

        ## Forward
        predictions = net(images)
        loss = loss_fn(predictions, labels)

        ## Backward
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()*labels.shape[0]
        labels_for_acc = np.concatenate((labels_for_acc, labels.cpu().numpy()), 0)
        preds_for_acc = np.concatenate((preds_for_acc, np.argmax(predictions.cpu().detach().numpy(), 1)), 0)

    accuracy = accuracy_score(labels_for_acc, preds_for_acc)
    return running_loss/TRAIN_SIZE, accuracy

def valid_fn(net, loader):
    
    running_loss = 0
    preds_for_acc = []
    labels_for_acc = []
    
    
    with torch.no_grad():       
        for _, (images, labels) in enumerate(loader):
            
            images, labels = images.to(device), labels.to(device)
            net.eval()
            predictions = net(images)
            loss = loss_fn(predictions, labels)
            
            running_loss += loss.item()*labels.shape[0]
            labels_for_acc = np.concatenate((labels_for_acc, labels.cpu().numpy()), 0)
            preds_for_acc = np.concatenate((preds_for_acc, np.argmax(predictions.cpu().detach().numpy(), 1)), 0)
            
        accuracy = accuracy_score(labels_for_acc, preds_for_acc)

    return running_loss/VALID_SIZE, accuracy


# Start training
leaf_model = model_resnet()
leaf_model.to(device)
optimizer = optim.Adam(leaf_model.parameters(), lr = LEARNING_RATE)

# Array to store loss and accuracy values
train_loss = []
valid_loss = []
train_acc = []
val_acc = []

if __name__ == "__main__":   
    for epoch in range(NUM_OF_EPOCH):
        tl, ta = train_fn(leaf_model, loader = train_loader)
        vl, va = valid_fn(leeaf_model, loader = valid_loader)
        train_loss.append(tl)
        valid_loss.append(vl)
        train_acc.append(ta)
        val_acc.append(va)
        
        if (epoch+1)%10==0:
            path = 'epoch' + str(epoch) + '.pt'
            torch.save(leaf_model.state_dict(), path)
        
        print('Epoch: '+ str(epoch) + ', Train loss: ' + str(tl) 
            + ', Val loss: ' + str(vl) + ', Train acc: ' + str(ta) + ', Val acc: ' + str(va))

