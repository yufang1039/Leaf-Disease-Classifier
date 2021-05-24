import math
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import models as models
from torch.utils.data import DataLoader
import time

from model import model_resnet
from dataset import LeafDataset
from sklearn.metrics import accuracy_score



# Define constant param
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_OF_EPOCH = 8
LEARNING_RATE = 8e-4
BATCH_SIZE = 8

# File directories
label_dir = "train_df.csv"
image_dir = "train_images/"
save_dir = "train_results/exp3/"

# Get leaf dataset and dataloader for both training and validation dataset
leaf_ds = LeafDataset(csv_file=label_dir, imgs_path=image_dir,
                transform=torchvision.transforms.Resize([500, 500]))
ds_len = leaf_ds.__len__()

TRAIN_SIZE = math.floor(ds_len)
VALID_SIZE = ds_len - math.floor(ds_len * 0.7)

# train_ds, valid_ds = torch.utils.data.random_split(dataset=leaf_ds, 
#                     lengths=[TRAIN_SIZE, VALID_SIZE])

# train_loader = DataLoader(train_ds, shuffle=True, batch_size=BATCH_SIZE)
# valid_loader = DataLoader(valid_ds, shuffle=True, batch_size=BATCH_SIZE)

train_loader = DataLoader(leaf_ds, shuffle=True, batch_size=BATCH_SIZE)

loss_fn = torch.nn.BCEWithLogitsLoss()

print_stats = True
## Train funciton that train the model for one epoch
def train_fn(net, loader):

    tr_loss = 0
    tr_accuracy = 0
    batch_number = 0

    for batch_number, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        ## Set gradient to zero
        optimizer.zero_grad()

        ## Forward
        predictions = net(images)
        loss = loss_fn(predictions, labels.squeeze(-1))

        ## Clean up Gradients
        net.zero_grad()

        ## Backward
        loss.backward()
        tr_loss += loss.item()

        ## Accuracy

        # batch_shape = list(predictions.size())
        # for i in range(batch_shape[0]):
        #     for j in range(batch_shape[1]):
        #         prediction = 1 if predictions.detach().cpu().numpy()[i][j] >= 0.5 else 0
        #         if prediction == labels.detach().cpu().numpy()[i][j]:
        #             tr_accuracy += 1.0/batch_shape[1]

        # print(tr_accuracy)

        positives = torch.zeros(predictions.size())
        positives[torch.nonzero(predictions > 0.5, as_tuple=True)] = 1
        tr_accuracy += torch.count_nonzero((positives - labels) == 0) / predictions.size()[1]

        optimizer.step()

        elapsed_time = time.time() - start
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)


        if print_stats:
            print("Current batch: " + str(batch_number+1) + "/" + str(int(TRAIN_SIZE/BATCH_SIZE)) \
                 + " , running loss is " + str(tr_loss/((batch_number+1) * BATCH_SIZE)) \
                + " , elapsed_time = " + "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds) \
                + " , epoch = " + str(epoch))

    return tr_accuracy/TRAIN_SIZE, tr_loss/TRAIN_SIZE

def valid_fn(net, loader):
    
    valid_loss = 0
    valid_accuracy = 0
    
    with torch.no_grad():       
        for _, (images, labels) in enumerate(loader):
            
            images, labels = images.to(device), labels.to(device)
            net.eval()
            predictions = net(images)
            loss = loss_fn(predictions, labels.squeeze(-1))
            
            valid_loss += loss.item()

            positives = torch.zeros(predictions.size())
            positives[torch.nonzero(predictions > 0.5, as_tuple=True)] = 1
            valid_accuracy += torch.count_nonzero((positives - labels) == 0) / predictions.size()[1]
            

    return valid_accuracy/VALID_SIZE, valid_loss/VALID_SIZE




# Start training
leaf_model = model_resnet()
leaf_model.to(device)
optimizer = optim.Adam(leaf_model.parameters(), lr=LEARNING_RATE)

# Array to store loss and accuracy values
train_loss = []
train_acc = []
valid_loss = []
valid_acc = []
train_acc = []
val_acc = []

if __name__ == "__main__":

    global start
    global epoch

    # Training Loop
    for epoch in range(NUM_OF_EPOCH):

        os.system(f'echo \"  Epoch {epoch}\"')
        
        ## Set mode to training
        leaf_model.train()

        start = time.time()

        ta, tl = train_fn(leaf_model, loader=train_loader)
        #va, vl = valid_fn(leaf_model, loader=valid_loader)
        train_loss.append(tl)
        #valid_loss.append(vl)
        train_acc.append(ta)
        #valid_acc.append(va)

        print('Epoch: '+ str(epoch) + ', Train loss: ' + str(tl) + ', Train accuracy: ' + str(ta) )
            # + ', Val loss: ' + str(vl) + ', Val accuracy: ' + str(va))

        if epoch % 20 == 0:
            torch.save(leaf_model.state_dict(), save_dir + str(epoch) + ".pt")

    # Saves model weights, need to specify file name
    torch.save(leaf_model.state_dict(), save_dir + "final.pt")

    # Plots Loss
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, 'y', label='Training loss')
    #plt.plot(epochs, valid_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_dir + "loss.jpg")
    plt.show()

    plt.plot(epochs, train_acc, 'g', label='Training accuracy')
    #plt.plot(epochs, valid_acc, 'b', label='Valid accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(save_dir + "accuracy.jpg")
    plt.show()
