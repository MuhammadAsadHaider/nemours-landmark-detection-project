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

target_labels = ['right femur head', 'right femur neck', 'left femur head', 'left femur neck']
target_colors = ['r', 'b', 'g', 'y']

image_paths = [item['image'] for item in data]
targets = [item['labels'] for item in data] 
num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
for fold, (train_indices, test_indices) in enumerate(kfold.split(image_paths)):
    test_image_paths = [image_paths[i] for i in test_indices]
    test_targets = [targets[i] for i in test_indices]
    test_dataset = CustomDataset(test_image_paths, test_targets, transform=transform)

    dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    if fold == 0:
        model.load_state_dict(torch.load(f"model_fold0_epoch188.pth"))
    elif fold == 1:
        model.load_state_dict(torch.load(f"model_fold1_epoch66.pth"))
    elif fold == 2:
        model.load_state_dict(torch.load(f"model_fold2_epoch1.pth"))
    elif fold == 3:
        model.load_state_dict(torch.load(f"model_fold3_epoch0.pth"))
    elif fold == 4:
        model.load_state_dict(torch.load(f"model_fold4_epoch4.pth"))

    model.eval()

    for image, target in tqdm(dataloader_test):
        image = image.to(device)
        target = target.to(device)
        with torch.no_grad():
            pred = model(image) # 32 x 16
            # plot the ounding boxes and save image
            for i in range(len(image)):
                # pred_boxes = pred[i].cpu().numpy() # 16
                # pred_boxes = pred_boxes.reshape(4, 4)
                # pred_boxes = pred_boxes.tolist()

                pred_boxes = target[i].cpu().numpy() # 16
                pred_boxes = pred_boxes.reshape(4, 4)
                pred_boxes = pred_boxes.tolist()

                im = image[i].cpu().numpy()

                plt.imshow(im.transpose(1, 2, 0))
                for j in range(4):
                    plt.gca().add_patch(plt.Rectangle((pred_boxes[j][0] * 224 * 224 / 100, pred_boxes[j][1] * 224 * 224 / 100), pred_boxes[j][2] * 224 * 224 / 100, pred_boxes[j][3] * 224 * 224 / 100, fill=False, edgecolor=target_colors[j], linewidth=2))
                plt.savefig(f"results/fold{fold}_image{i}.png")
                plt.close()




