'''
Todo:
- Get the ViT working - done
- Get above 80% accuracy
- Get above 85% accuracy
- Get above 90% accuracy
- Get above 95% accuracy
'''

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
import torch.nn as nn
import torch.nn.functional as F

import wandb
import loss as losslib
import dataparsing

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 1) Load your pretrained VAE before defining Net
    vae = dataparsing.load_VAE(device)
    vae.eval()

    class Net(nn.Module):
        def __init__(self, VAE):
            super(Net, self).__init__()
            self.vae = VAE
            self.fc1 = nn.Linear(256, 512)
            self.fc2 = nn.Linear(512, 1024)
            self.fc3 = nn.Linear(1024, 256)
            self.fc4 = nn.Linear(256, 4)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            with torch.no_grad():
                mu, logvar = self.vae.encode(x)
                x = torch.cat([mu, logvar], dim=1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = F.relu(self.fc3(x))
            x = self.dropout(x)
            x = self.fc4(x)
            return x

    # 3) Instantiate classifier *with* the VAE
    model = Net(vae).to(device)

    # Initialize wandb inside main so that it is not re-run in child processes.
    wandb.init(project="radiowaves-classifier-VIT")
    wandb.config = {
        "learning_rate": 5e-4,
        "epochs": 20,
        "batch_size": 1
    }
    
    # Instantiate datasets and DataLoaders in the main process
    train_dataset = dataparsing.AugmentedDataset(dataparsing.train)#, transforms=True, vae=vae, device=device)
    test_dataset = dataparsing.AugmentedDataset(dataparsing.test)
    val_dataset = dataparsing.AugmentedDataset(dataparsing.val)
    
    train_loader = DataLoader(train_dataset, batch_size=wandb.config["batch_size"], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=wandb.config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=wandb.config["batch_size"], shuffle=True, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = schedulefree.adamw_schedulefree.AdamWScheduleFree(
        model.parameters(),
        lr=wandb.config["learning_rate"],
        weight_decay=1e-4
    )

    for epoch in range(wandb.config["epochs"]):
        model.train()
        running_loss = 0.0
        running_accuracy = 0
        total = 0
        running_unweighted_loss = 0.0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            optimizer.train()
            outputs = model(images)
            loss = losslib.CB_loss(labels, outputs, "normal", losslib.precomputed_weights)

            # Add explicit L2 regularization
            l2_lambda = 1e-4  # adjust as desired
            l2_reg = 0.0
            for param in model.parameters():
                l2_reg += torch.norm(param, 2)**2
            loss += l2_lambda * l2_reg

            running_unweighted_loss += criterion(outputs, labels).item()
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            running_loss += loss.item()
            running_accuracy += (predicted == labels).sum().item()
        
        wandb.log({
            "weighted loss": running_loss / len(train_loader), 
            "accuracy": running_accuracy / total, 
            "loss": running_unweighted_loss / len(train_loader)
        }, step=epoch)
        
        # Validation loop
        model.eval()
        optimizer.eval()
        val_loss = 0.0
        val_unweighted_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_unweighted_loss += loss.item()
                val_loss += losslib.CB_loss(labels, outputs, "normal", losslib.precomputed_weights)
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        wandb.log({
            "val_weighted_loss": val_loss / len(test_loader), 
            "val_accuracy": correct / total, 
            "val_loss": val_unweighted_loss / len(test_loader)
        }, step=epoch)

        save_path = f"classifierwithVAEencoder.pth"
        torch.save(model.state_dict(), save_path)