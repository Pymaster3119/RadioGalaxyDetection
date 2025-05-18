import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
import timm
import schedulefree.adamw_schedulefree

# Global variables that are safe (won't trigger side-effects on spawn)
imgs = []

# Define a transform object for augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=360),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
])

# Load data
with open("train_cropped_X.obj", 'rb') as f:
    with open("train_cropped_y.obj", 'rb') as g:
        train = (pickle.load(f), pickle.load(g))

with open("test_cropped_X.obj", 'rb') as f:
    with open("test_cropped_y.obj", 'rb') as g:
        test = (pickle.load(f), pickle.load(g))

with open("val_cropped_X.obj", 'rb') as f:
    with open("val_cropped_y.obj", 'rb') as g:
        val = (pickle.load(f), pickle.load(g))

class AugmentedDataset(Dataset):
    def __init__(self, data, transforms=False):
        self.data = data
        self.transforms = transforms
        
    def __len__(self):
        return len(self.data[0]) * 10 if self.transforms else len(self.data[0])
        
    def __getitem__(self, idx):
        idx = idx % len(self.data[0])
        X_cropped = self.data[0][idx].astype(np.float32)
        X_cropped = np.transpose(X_cropped, (2, 0, 1))
        selected_channel = self.data[1][idx]
        if self.transforms:
            # Apply transforms (on CPU) and add noise
            X_cropped = transform(torch.tensor(X_cropped)).numpy()
            noise = np.random.normal(0, 0.1, X_cropped.shape).astype(np.float32)
            X_cropped = X_cropped + noise
            X_cropped = np.clip(X_cropped, 0, 1)
        return X_cropped, selected_channel

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")