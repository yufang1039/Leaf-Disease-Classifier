import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader

from model import model_resnet
from dataset import LeafDataset

# Define constant param
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_OF_EPOCH = 2
BATCH_SIZE = 8

# File directories
label_dir = "train_df_small.csv"
image_dir = "train_images/"
save_dir = "test_results/"
weights_path = "train_results/final.pt"


# Get leaf dataset and dataloader for both training and validation dataset
leaf_ds = LeafDataset(csv_file=label_dir, imgs_path=image_dir,
                      transform=torchvision.transforms.Resize([500, 500]))
TEST_SIZE = leaf_ds.__len__()

test_loader = DataLoader(leaf_ds, shuffle=True, batch_size=BATCH_SIZE)

## Train funciton that train the model for one epoch

def test_fn(net, loader):
    test_accuracy = 0

    with torch.no_grad():
        for _, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            predictions = net(images)

            batch_shape = list(predictions.size())
            for i in range(batch_shape[0]):
                for j in range(batch_shape[1]):
                    prediction = 1 if predictions.detach().cpu().numpy()[i][j] >= 0.5 else 0
                    if prediction == labels.detach().cpu().numpy()[i][j]:
                        test_accuracy += 1.0 / batch_shape[1]

    return test_accuracy / TEST_SIZE

# Start training
leaf_model = model_resnet()
leaf_model.to(device)

# Array to store loss and accuracy values
test_acc = []

if __name__ == "__main__":
    model = model_resnet()
    model.to(device)
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    for epoch in range(NUM_OF_EPOCH):
        ta = test_fn(leaf_model, loader=test_loader)
        test_acc.append(ta)
        if (epoch + 1) % 10 == 0:
            print('Epoch: ' + str(epoch) + ' Test accuracy: ' + str(ta))

    print("Avg Test accuracy: " + str(sum(test_acc)/len(test_acc)))


    # Plots Loss
    epochs = range(1, len(test_acc) + 1)
    plt.plot(epochs, test_acc, 'b', label='Test accuracy')
    plt.title('Test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(save_dir + "test_accuracy.jpg")
    plt.show()




