# %%
# !pip install einops
# !pip install monai==1.3.2
# !pip install rasterio

# %%
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
import pandas as pd
from sklearn.impute import SimpleImputer
import torch
from sklearn.model_selection import KFold
from monai.networks.nets import SwinUNETR, DynUNet, UNet
from monai.transforms import RandFlipd, RandRotate90d
from torch import nn
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score
import torch.optim as optim
from tqdm import tqdm
import gc
import rasterio
from monai.losses import DiceLoss
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms.v2 import GaussianNoise
import random
from monai.inferers import SlidingWindowInferer
import argparse

# %%
class CustomImageDataset(Dataset):
    def __init__(self, data_list, transform=True):
        self.transform   = transform
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.fromarray(self.data[idx][:4].transpose([1, 2, 0])).convert('RGB')
        image = transforms.ToTensor()(image)
        label    = torch.tensor(self.data[idx][4:])
        if self.transform:
            image_label = {'image': image, 'label': label}
            image_label = RandFlipd(keys=['image', 'label'], spatial_axis=1)(image_label)
            image_label = RandFlipd(keys=['image', 'label'], spatial_axis=0)(image_label)
            image_label = RandRotate90d(keys=['image', 'label'], spatial_axes=(0, 1))(image_label)
            image, label = image_label['image'], image_label['label']
            if random.random() < 0.1:
                image = GaussianNoise(sigma=0.01)(image)
            if random.random() < 0.1:
                image = transforms.GaussianBlur(5)(image)
            if random.random() < 0.1:
                image = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)(image)         
            
        return image, label

# %%
def create_dataloaders(data_list, batch_size=32, n_fold=0):
    # Initialize dataset
    dataset = CustomImageDataset(data_list=data_list, transform=True)
    
    # Create train/validation split
    kf = KFold(n_splits=5, shuffle=True, random_state=2024)
    for i, (train_index, val_index) in enumerate(kf.split(dataset)):
        if i == n_fold:
            break
            
    train_dataset = Subset(dataset, train_index)
    dataset = CustomImageDataset(data_list=data_list, transform=False)
    val_dataset = Subset(dataset, val_index)
    print(len(train_dataset))
    print(len(val_dataset))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader

# %%
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.dice = DiceLoss(to_onehot_y=True, softmax=True, batch=True)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, p, y):
#         return self.ce(p, y[:, 0, ...]) + self.dice(p, y)
        return self.ce(p, y[:, 0, ...])

# %%
def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, deep_supervision=False):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device).float(), labels.to(device).long()

        optimizer.zero_grad()
        
        outputs = model(images)
        if deep_supervision:
            major_loss = criterion(outputs[:, 0, ...], labels)
            moderate_loss = criterion(outputs[:, 1, ...], labels)
            minor_loss = criterion(outputs[:, 2, ...], labels)
            loss = major_loss + 0.5 * moderate_loss + 0.25 * minor_loss
        else:
            loss = criterion(outputs, labels)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_true = 0.0
    running_true_false = 0.0
#     all_labels = []
#     all_outputs = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device).float(), labels.to(device).long()

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            outputs = torch.softmax(outputs, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            preds = outputs.argmax(axis=1, keepdims=True)
            running_true += accuracy_score(labels.reshape(-1), preds.reshape(-1), normalize=False) * 0.6 + accuracy_score((labels == 0).astype(int).reshape(-1), (preds == 0).astype(int).reshape(-1), normalize=False) * 0.4
            running_true_false  += len(labels.reshape(-1))
#             all_labels.append(labels.cpu().numpy())
#             all_outputs.append(outputs.cpu().numpy())

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_metric = running_true / running_true_false
    
#     all_labels = np.concatenate(all_labels)
#     all_outputs = np.concatenate(all_outputs)
#     all_outputs = torch.softmax(torch.tensor(all_outputs), dim=1).numpy()  # Convert logits to probabilities
    
    return epoch_loss, epoch_metric

# %%
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

# %%
def train_model(data_list, model, model_name, num_epochs=10, batch_size=32, lr=1e-4, n_fold=0, device='cuda', patience=3, warmup_epochs=0, 
               deep_supervision=False):
    train_loader, val_loader = create_dataloaders(data_list=data_list, batch_size=batch_size, n_fold=n_fold)
    train_sets = len(train_loader)

    model     = model.to(device)
    criterion = Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, train_sets * warmup_epochs, train_sets * num_epochs)
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    train_losses = []
    val_losses   = []
    
    path = model_name + str(n_fold)
    os.makedirs(path, exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, deep_supervision)
        val_loss, accuracy = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Calculate metrics on validation set
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.4f}')
        early_stopping(-accuracy, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
        
    # Plot Loss
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.savefig(f'loss_plot{n_fold}.png')
    plt.show()

# %% [markdown]
# # TODO  
# - data agumentation  
# - deep supervision
# - test time agumentation

# %%
data = np.load('dataset/features_label.npy')
gc.collect()
parser = argparse.ArgumentParser()
parser.add_argument('--n_fold', default=0, type=int)
args = parser.parse_args()
model = DynUNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=3,
    kernel_size=(3, 3, 3, 3, 3, 3, 3),
    strides=(1, 2, 2, 2, 2, 2, 2),
    upsample_kernel_size=(2, 2, 2, 2, 2, 2),
    deep_supervision=True, 
    deep_supr_num=2, 
#         dropout=0.0, 
    filters=[64, 96, 128, 192, 256, 384, 512]
)
n_fold = args.n_fold
batch_size = 128
lr = 2e-3
epochs = 50
train_model(data, model, 'DynUNet', num_epochs=epochs, batch_size=batch_size, 
            lr=lr, n_fold=n_fold, device='cuda', patience=5, warmup_epochs=3, deep_supervision=True)
torch.cuda.empty_cache()
gc.collect()

# %%



