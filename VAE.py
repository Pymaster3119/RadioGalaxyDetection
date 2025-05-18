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

import wandb
import matplotlib.pyplot as plt
import loss as losslib
import dataparsing

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

if __name__ == '__main__':
    wandb.init(project="radiowaves-VAE")
    wandb.config = {
        "learning_rate": 1e-3,
        "epochs": 50,
        "batch_size": 32,
        "latent_dim": 128,
        "beta": 1.0,
    }

    # Instantiate datasets and DataLoaders in the main process
    train_dataset = dataparsing.AugmentedDataset(dataparsing.train, transforms=True)
    test_dataset = dataparsing.AugmentedDataset(dataparsing.test)
    val_dataset = dataparsing.AugmentedDataset(dataparsing.val)

    train_loader = DataLoader(train_dataset, batch_size=wandb.config["batch_size"], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=wandb.config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=wandb.config["batch_size"], shuffle=True, num_workers=4)

    # Instantiate the model
    vae = VAE(latent_dim=wandb.config['latent_dim']).to(device)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=wandb.config["learning_rate"])

    

    # Training loop
    for epoch in range(wandb.config["epochs"]):
        vae.train()
        running_loss = 0.0
        total_samples = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{wandb.config['epochs']} - Training"):
            # For compatibility if batch is a list of tensors
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = vae(inputs)
            loss = losslib.VAE_loss_function(recon_x, inputs, mu, logvar, beta=wandb.config["beta"])
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
        
        avg_train_loss = running_loss / total_samples
        wandb.log({"Train Loss": avg_train_loss}, step=epoch)

        # Validation loop
        vae.eval()
        running_val_loss = 0.0
        total_val_samples = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{wandb.config['epochs']} - Validation"):
                inputs = batch[0].to(device)
                recon_x, mu, logvar = vae(inputs)
                loss = losslib.VAE_loss_function(recon_x, inputs, mu, logvar, beta=wandb.config["beta"])
                running_val_loss += loss.item() * inputs.size(0)
                total_val_samples += inputs.size(0)
        
        avg_val_loss = running_val_loss / total_val_samples
        wandb.log({"Validation Loss": avg_val_loss}, step=epoch)
        # Log a sample input and its reconstruction as an image to WANDB
        vae.eval()
        sample_batch = next(iter(val_loader))
        sample_input = sample_batch[0].to(device)
        with torch.no_grad():
            sample_output, _, _ = vae(sample_input)
        sample_input = sample_input.cpu().numpy()
        sample_output = sample_output.cpu().numpy()

        num_samples = min(8, sample_input.shape[0])
        fig, axs = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
        for i in range(num_samples):
            # Input image
            axs[0, i].imshow(sample_input[i].transpose(1, 2, 0))
            axs[0, i].axis("off")
            # Reconstructed image
            axs[1, i].imshow(sample_output[i].transpose(1, 2, 0))
            axs[1, i].axis("off")
        wandb.log({"Input-Reconstruction": wandb.Image(fig)}, step=epoch)
        plt.close(fig)