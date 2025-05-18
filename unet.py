import pickle
from collections import namedtuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import random
import wandb
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import pickle


assert(torch.cuda.is_available())

# Initialize wandb
wandb.init(project="radiowaves-unet")
wandb.config = {
    "learning_rate": 0.001,
    "epochs": 25,
    "batch_size": 32
}

imgs = []
annotationclass = namedtuple('annotationclass', ['bbox','category','keypoints','segmentation'])
imageclass = namedtuple('imageclass', ['image', 'normalized', 'annotations', 'mask'])

with open("train.obj", 'rb') as f:
    train = pickle.load(f)

with open("test.obj", 'rb') as f:
    test = pickle.load(f)

with open("val.obj", 'rb') as f:
    val = pickle.load(f)

class AugmentedDataset(Dataset):
    def __init__(self, data, transforms=False):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        X = sample.normalized.astype(np.float32)
        Y = sample.mask.astype(np.float32)
        
        if self.transforms:
            if random.random() < 0.5:
                X = np.fliplr(X).copy()
                Y = np.fliplr(Y).copy()
            if random.random() < 0.5:
                X = np.flipud(X).copy()
                Y = np.flipud(Y).copy()

        X = X[:448, :448, :]
        Y = Y[:448, :448, :]
        # C,W,H
        X = np.transpose(X, (2, 0, 1))
        Y = np.transpose(Y, (2, 0, 1))
        return X, Y



train_dataset = AugmentedDataset(train, transforms=True)
test_dataset = AugmentedDataset(test)
val_dataset = AugmentedDataset(val)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)


# # Get one batch of data from train_loader
# data_iter = iter(train_loader)
# images, masks = next(data_iter)

# # Print the shapes of the images and masks
# print(images.shape, masks.shape)
# import matplotlib.pyplot as plt

# # Display the first 32 images and their masks
# while True:
#     images, masks = next(data_iter)
#     fig, axes = plt.subplots(4, 8)
#     for x in range(4):
#         for y in range(8):
#             i = x * 8 + y
#             print(masks[i].numpy().shape)
#             array = np.sum(masks[i].numpy(), axis=2)
#             array = np.repeat(array[:, :, np.newaxis], 3, axis=2)
#             axes[x, y].imshow((images[i].numpy() + array)/2, cmap='gray')
#             axes[x, y].axis('off')

#     plt.tight_layout()
#     plt.show()

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   

def dice_loss(pred, target, smooth = 1.):
    pred = F.sigmoid(pred)
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
                
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(n_class=4).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 15

def focal_loss(inputs, targets, alpha=0.5, gamma=2):
    # Binary Cross-Entropy loss calculation
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    pt = torch.exp(-bce_loss)  # Convert BCE loss to probability
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss  # Apply focal adjustment
    return focal_loss.mean()

best_val_loss = 1e10
print(model)
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(train_loader, "Training"):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = focal_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss}")
    wandb.log({"epoch": epoch+1, "train_loss": avg_train_loss})
    
    # Validation loop
    model.eval()
    val_loss = 0.0
    dice_loss_val = 0.0
    with torch.no_grad():
        for images, masks in tqdm(test_loader, "Validation"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = focal_loss(outputs, masks)
            dice_loss_val += dice_loss(outputs, masks)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {avg_val_loss}, Dice Loss: {dice_loss_val/len(val_loader)}")
    wandb.log({"epoch": epoch+1, "val_loss": avg_val_loss, "dice_loss": dice_loss_val/len(val_loader)})

    # Save the model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        print(f"NEW PR!!! {best_val_loss}")
        torch.save(model.state_dict(), f"unet_epoch.pth")