import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
from tqdm import tqdm
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np


class ResNet(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes * 4)  # Four coordinates for each class

    def forward(self, x):
        x = self.resnet(x)
        return x


# Create an instance of your ResNet model
model = ResNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define your loss function and optimizer
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
])

def calculate_iou(gt, pred):
    # gt: ground truth
    # pred: prediction
    # gt and pred should be numpy arrays of shape (4,)

    # convert bounding boxes to the correct format
    # (x1, y1, x2, y2)
    gt = [gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3]]
    pred = [pred[0], pred[1], pred[0] + pred[2], pred[1] + pred[3]]

    # calculate the intersection points
    x1 = max(gt[0], pred[0])
    y1 = max(gt[1], pred[1])
    x2 = min(gt[2], pred[2])
    y2 = min(gt[3], pred[3])

    # calculate the area of intersection rectangle
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # calculate the area of both the prediction and ground truth rectangles
    gt_area = (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1)

    pred_area = (pred[2] - pred[0] + 1) * (pred[3] - pred[1] + 1)

    # calculate the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(gt_area + pred_area - intersection_area)

    return iou


class CustomDataset(Dataset):
    def __init__(self, image_paths, targets, transform=None):
        self.image_paths = image_paths
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        target = self.targets[index]

        # Load the image
        image = Image.open(image_path).convert("RGB")

        # Apply transformations if specified
        if self.transform is not None:
            image = self.transform(image)

        return image, target


with open("targets.json") as f:
    data = json.load(f)


image_paths = [item['image'] for item in data]
targets = [item['labels'] for item in data] 
num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
for fold, (train_indices, test_indices) in enumerate(kfold.split(image_paths)):
    train_image_paths = [image_paths[i] for i in train_indices]
    test_image_paths = [image_paths[i] for i in test_indices]
    train_targets = [targets[i] for i in train_indices]
    test_targets = [targets[i] for i in test_indices]


    train_dataset = CustomDataset(train_image_paths, train_targets, transform=transform)
    test_dataset = CustomDataset(test_image_paths, test_targets, transform=transform)

    dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

    num_epochs = 200

    train_losses = []
    test_losses = []
    test_ious = []
    train_epoch_losses = []
    test_epoch_losses = []
    test_epoch_ious = []
    lowest_test_loss = 100000000

    for epoch in range(num_epochs):
        model.train()
        train_running_loss = 0.0

        for images, labels in tqdm(dataloader_train):
            images = images.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Reshape the output to match the target shape
            outputs = outputs.reshape(-1, 4)

            # Reshape the targets to match the output shape
            labels = labels.reshape(-1, 4)

            # Calculate the loss
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()
            train_losses.append(loss.item())

        
        # calculate test loss
        model.eval()
        test_running_loss = 0.0
        for images, labels in tqdm(dataloader_test):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Reshape the output to match the target shape
            outputs = outputs.reshape(-1, 4)

            # Reshape the targets to match the output shape
            labels = labels.reshape(-1, 4)

            # Calculate the loss
            loss = criterion(outputs, labels)

            # calculate the iou
            batch_ious = []
            for i in range(len(outputs)):
                iou = calculate_iou(labels[i].detach().cpu().numpy(), outputs[i].detach().cpu().numpy())
                batch_ious.append(iou)
            test_ious.append(np.mean(batch_ious))

            test_running_loss += loss.item()
            test_losses.append(loss.item())

        train_epoch_loss = train_running_loss / len(dataloader_train)
        train_epoch_losses.append(train_epoch_loss)
        test_epoch_loss = test_running_loss / len(dataloader_test)
        test_epoch_losses.append(test_epoch_loss)
        test_epoch_iou = np.mean(test_ious)
        test_epoch_ious.append(test_epoch_iou)

        if test_epoch_loss < lowest_test_loss:
            lowest_test_loss = test_epoch_loss
            torch.save(model.state_dict(), f"model_fold{fold}_epoch{epoch}.pth")
    

    # save the losses as numpy arrays
    np.save(f"train_loss_{fold}.npy", train_epoch_losses)
    np.save(f"test_loss_{fold}.npy", test_epoch_losses)
    np.save(f"train_loss_raw_{fold}.npy", train_losses)
    np.save(f"test_loss_raw_{fold}.npy", test_losses)
    np.save(f"test_ious_{fold}.npy", test_epoch_ious)
    np.save(f"test_ious_raw_{fold}.npy", test_ious)

    # plot and save the losses
    plt.plot(train_epoch_losses, label="train")
    plt.legend()
    plt.savefig(f"train_loss_{fold}.png")

    plt.plot(test_epoch_losses, label="test")
    plt.legend()
    plt.savefig(f"test_loss_{fold}.png")

    # plot the losses
    plt.plot(train_losses, label="train")
    plt.legend()
    plt.savefig(f"train_loss_raw_{fold}.png")

    plt.plot(test_losses, label="test")
    plt.legend()
    plt.savefig(f"test_loss_raw_{fold}.png")

    # plot the ious
    plt.plot(test_epoch_ious, label="test")
    plt.legend()
    plt.savefig(f"test_ious_{fold}.png")

    plt.plot(test_ious, label="test")
    plt.legend()
    plt.savefig(f"test_ious_raw_{fold}.png")

    




    





