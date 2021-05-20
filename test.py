from typing import final
import torch
import torchvision
import pandas as pd
from torch.utils.data import DataLoader

from model import model_resnet
from dataset import TestDataSet

# Define constant param
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_OF_EPOCH = 5
BATCH_SIZE = 8

# File directories
label_dir = "train_df_small.csv"
image_dir = "train_images/"
save_dir = "test_results/"
weights_path = "train_results/exp3/final.pt"

Categories = ["healthy", "scab", "frog_eye_leaf_spot", "rust", "complex", "powdery_mildew"]

if __name__ == "__main__":
    model = model_resnet()
    model.to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    submission = pd.DataFrame(columns=["image", "labels"])
    image_ids = []
    labels = []

    test_ds = TestDataSet(image_dir, transform=torchvision.transforms.Resize([500, 500]))
    test_loader = DataLoader(test_ds)
    for idx, (X, name) in enumerate(test_loader):
        y_pred = model(X)
        name = name[0]
        final_pred = ""
        for i in range(len(y_pred[0])):
            if y_pred[0][i] >= 0.5:
                final_pred += Categories[i] + " "
        final_pred = final_pred[:-1]
        image_ids.append(name)
        labels.append(final_pred)
    submission["image"] = image_ids
    submission["labels"] = labels

    print(submission) 
    submission.to_csv("submission.csv", index=False)






