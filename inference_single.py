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
from monai.transforms import RandFlipd, RandRotate90d, Flip, Compose
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
model = DynUNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=3,
    kernel_size=(3, 3, 3, 3, 3, 3, 3),
    strides=(1, 2, 2, 2, 2, 2, 2),
    upsample_kernel_size=(2, 2, 2, 2, 2, 2),
    deep_supervision=True, 
    deep_supr_num=2, 
    dropout=0.2, 
    filters=[64, 96, 128, 192, 256, 384, 512]
)

# %%
def predict(data, model, model_name, batch_size=32, n_fold=0, device='cuda', tta=False):
    model     = model.to(device) # load the model into the GPU
    model.load_state_dict(torch.load(os.path.join(model_name + str(n_fold), 'checkpoint.pth')))
    
    model.eval()
    with torch.no_grad():
        inferer = SlidingWindowInferer(roi_size=(128, 128), sw_batch_size=batch_size, overlap=0.5, mode="gaussian", progress=True, sw_device=device, device=torch.device('cpu'))
        outputs = inferer(data, model)
        outputs = torch.softmax(outputs, dim=1)
    if tta:
        tta_list = [Flip(spatial_axis=0), Flip(spatial_axis=1), Compose([Flip(spatial_axis=0), Flip(spatial_axis=1)])]
        tta_res = [outputs]
        for aug in tta_list:
            with torch.no_grad():
                inferer = SlidingWindowInferer(roi_size=(128, 128), sw_batch_size=batch_size, overlap=0.5, mode="gaussian", progress=True, sw_device=device, device=torch.device('cpu'))
                transformed_data = aug(data[0]).unsqueeze(0)
                outputs = inferer(transformed_data, model)
                outputs = aug.inverse(outputs[0]).unsqueeze(0)
                outputs = torch.softmax(outputs, dim=1)
                tta_res.append(outputs)
            gc.collect()
        outputs = torch.stack(tta_res, dim=0).mean(dim=0)

    return outputs

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--tta', default=False, action='store_true')
args = parser.parse_args()
tta = args.tta
with rasterio.open('dataset/result.tif') as src:
    data = src.read()[:3, ...]
print(data.shape)
gc.collect()
preds_i = []
batch_size = 128
step_size = 10000
for i in range(0, data.shape[1], step_size):
    preds_j = []
    for j in range(0, data.shape[2], step_size):
        start_i = i
        end_i = i + step_size if i + step_size <= data.shape[1] else data.shape[1]
        start_j = j
        end_j = j + step_size if j + step_size <= data.shape[2] else data.shape[2]
        chunk_data = data[:, start_i: end_i, start_j: end_j]     
        chunk_data = torch.tensor(chunk_data) / 255.0
        fold_preds = []
        for n_fold in range(1):
            preds = predict(chunk_data.float().unsqueeze(0), model, 'DynUNet', batch_size, n_fold, 'cuda', tta=tta).numpy()
            fold_preds.append(preds)
        preds = np.stack(fold_preds, axis=0).mean(axis=0)
        preds = preds.argmax(axis=1).astype(np.uint8)
        preds_j.append(preds)
        gc.collect()
    preds_j = np.concatenate(preds_j, axis=2)
    preds_i.append(preds_j)
    gc.collect()
preds = np.concatenate(preds_i, axis=1)
del data
del preds_i
del preds_j
gc.collect()

# %%
with rasterio.open('dataset/result.tif') as src:
    mask = src.read()[-1:, ...]
preds = np.where(mask == 0, 0, preds)
del mask
gc.collect()

# %%
with rasterio.open('dataset/result.tif') as src:
    meta = src.meta.copy()
print(meta)
meta['count'] = 1
meta['nodata'] = None
path = 'pred_tta.tif' if tta else 'pred_single.tif'
with rasterio.open(path, "w", **meta) as dst:
    dst.write(preds)

# %%



