import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import math
import re

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
        image = torchvision.io.read_image(image_path) 
        target = torch.tensor(row[-6:], dtype=torch.float32)
        if self.transform:
            return self.transform(image), target
        return image, target

