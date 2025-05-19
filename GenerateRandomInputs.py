import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
import timm

import wandb
import matplotlib.pyplot as plt
import loss as losslib
import dataparsing

# Set device
device = torch.device("mps")#torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
if __name__ == "__main__":
    # Load the model
    model = VAE(128).to(device)
    # Instantiate datasets and DataLoaders in the main process
    train_dataset = dataparsing.AugmentedDataset(dataparsing.train, transforms=True)
    test_dataset = dataparsing.AugmentedDataset(dataparsing.test)
    val_dataset = dataparsing.AugmentedDataset(dataparsing.val)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4)

    # Load the saved model weights
    model = torch.load('VAEmodel.pth', map_location=device)
    model.eval()

    #Run on one input
    batch = next(iter(test_loader))
    input = batch[0].to(device)         # shape: [1,3,H,W]

    # squeeze batch dim for both input and outputs
    img = input.squeeze(0).cpu().numpy()  

    outputs = []
    for i in range(10):
        with torch.no_grad():
            recon, mu, logvar = model(input)
            outputs.append(recon.squeeze(0).cpu().numpy())
    
    #Plot the input and outputs with matplotlib
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    axs = axs.flatten()
    axs[0].imshow(img.transpose(1, 2, 0))
    axs[0].set_title("Input")
    for i in range(1, 10):
        axs[i].imshow(outputs[i-1].transpose(1, 2, 0))
        axs[i].set_title(f"Output {i}")
    plt.tight_layout()
    plt.show()