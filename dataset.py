import torch
import torchvision
from torch.utils.data import Dataset
import pandas as pd
import os

class LeafDataset(Dataset):
    
    def __init__(self, csv_file, imgs_path, transform=None):
        self.df = pd.read_csv(csv_file) 
        self.imgs_path = imgs_path 
        self.transform = transform 
        self.len = self.df.shape[0] 
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index): 
        row = self.df.iloc[index]
        image_path = self.imgs_path + row[0]
        image = torchvision.io.read_image(image_path).float()
        target = torch.tensor(row[-6:], dtype=torch.float)
        if self.transform:
            return self.transform(image), target
        return image, target



class TestDataSet(Dataset):

    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.total_imgs = os.listdir("test_images/")

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        name = self.total_imgs[idx]
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = torchvision.io.read_image(img_loc).float()
        if self.transform:
            return self.transform(image), name
        return image, name