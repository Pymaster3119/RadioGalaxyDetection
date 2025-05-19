import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
import timm
import random

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
with open("/content/drive/MyDrive/Colab Notebooks/Radio Waves/train_cropped_X.obj", 'rb') as f:
    with open("/content/drive/MyDrive/Colab Notebooks/Radio Waves/train_cropped_y.obj", 'rb') as g:
        train = (pickle.load(f), pickle.load(g))

with open("/content/drive/MyDrive/Colab Notebooks/Radio Waves/test_cropped_X.obj", 'rb') as f:
    with open("/content/drive/MyDrive/Colab Notebooks/Radio Waves/test_cropped_y.obj", 'rb') as g:
        test = (pickle.load(f), pickle.load(g))

with open("/content/drive/MyDrive/Colab Notebooks/Radio Waves/val_cropped_X.obj", 'rb') as f:
    with open("/content/drive/MyDrive/Colab Notebooks/Radio Waves/val_cropped_y.obj", 'rb') as g:
        val = (pickle.load(f), pickle.load(g))

class AugmentedDataset(Dataset):
    def __init__(self, data, transforms=False, vae = False, device = None):
        self.data = data
        self.transforms = transforms
        self.vae = vae
        self.device = device
    def __len__(self):
        return len(self.data[0]) * 10 if self.transforms else len(self.data[0])
        
    def __getitem__(self, idx):
        idx = idx % len(self.data[0])
        X_cropped = self.data[0][idx].astype(np.float32)
        X_cropped = np.transpose(X_cropped, (2, 0, 1))
        selected_channel = self.data[1][idx]
        if self.transforms:
            # Apply CPU‐side augmentations
            X_cropped = transform(torch.tensor(X_cropped)).numpy()
            noise = np.random.normal(0, 0.1, X_cropped.shape).astype(np.float32)
            X_cropped = np.clip(X_cropped + noise, 0, 1)

            if self.vae: #and random.random() < 0.5:
                # Prepare a batch dim, send to device, run VAE
                vae_input = torch.tensor(X_cropped, dtype=torch.float32).unsqueeze(0).to(self.device)
                recon, mu, sigma = self.vae(vae_input)
                # bring back to CPU→numpy and remove batch dim
                recon = recon.squeeze(0).cpu().detach().numpy()
                X_cropped = np.clip(recon, 0, 1)

        return X_cropped, selected_channel

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1), 
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(x.size(0), 256, 4, 4)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def load_VAE(device):
    # Set device
    model = VAE(128).to(device)

    # Load the saved model weights
    model = torch.load('/content/drive/MyDrive/Colab Notebooks/Radio Waves/VAEmodel.pth', map_location=device, weights_only = False)
    model.eval()

    return model