# 2021-05-25
# Test script based on the analysis_script.ipynb

import numpy as np
import pandas as pd
from pathlib import Path
import os, sys, datetime, time, random, fnmatch, math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import skimage.metrics

import torch
from torchvision import transforms as tvtransforms
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.utils as vutils
import torch.utils.tensorboard as tensorboard
import torch.nn as nn

import datasets, transforms, GusarevModel, pytorch_msssim

# Flags:
flag_savePictures = False

# Path to the trained network
PATH_SAVE_NETWORK_INTERMEDIATE = "./trained_network.tar"


# Data
# To run your own data, set up a PyTorch dataset in the datasets.py file
# The network was trained/tested on the "internal" dataset.
_batch_size = 10
image_spatial_size = (440,440)
switch = "external_POLYU" #"internal"##

if switch == "internal":
    directory_source = "D:/data/JSRT/augmented/test/source/"
    directory_boneless = "D:/data/JSRT/augmented/test/target/"
    keys_images = ["source", "boneless"]
    ds = datasets.JSRT_CXR(directory_source, directory_boneless, 
                           transform=tvtransforms.Compose([
                                 transforms.RescalingNormalisation(keys_images,(0,1)),
                                 transforms.RandomIntensityComplement(keys_images, probability=0.5),
                                 transforms.Rescale(image_spatial_size, keys_images, None),
                                 transforms.ToTensor(keys_images),
                                 ]))
elif switch == "external_POLYU":
    externalTest_directory = "D:/data/POLYU_COVID19_CXR_CT_Cohort1/cxr/CXR_PNG"
    keys_images = ["source"]
    ds = datasets.POLYU_COVID19_CXR_CT_Cohort1(externalTest_directory,
                                 transform=tvtransforms.Compose([
                                 transforms.RescalingNormalisation(keys_images,(0,1)),
                                 transforms.RandomIntensityComplement(keys_images, probability=0.5),
                                 transforms.Rescale(image_spatial_size, keys_images, None),
                                 transforms.ToTensor(keys_images),
                                 ]))
print(len(ds))
dl = DataLoader(ds, _batch_size, shuffle=True, num_workers=0)


## NETWORK
# Load network from the trained_network.tar file
input_array_size = (_batch_size, 1, image_spatial_size[0], image_spatial_size[1])
net = GusarevModel.MultilayerCNN(input_array_size)
#net = nn.DataParallel(net, list(range(ngpu)))
if os.path.isfile(PATH_SAVE_NETWORK_INTERMEDIATE):
    print("=> loading checkpoint '{}'".format(PATH_SAVE_NETWORK_INTERMEDIATE))
    checkpoint = torch.load(PATH_SAVE_NETWORK_INTERMEDIATE, map_location='cpu')
    start_epoch = checkpoint['epoch_next']
    reals_shown_now = checkpoint['reals_shown']
    net.load_state_dict(checkpoint['model_state_dict'])
    print("=> loaded checkpoint '{}' (epoch {}, reals shown {})".format(PATH_SAVE_NETWORK_INTERMEDIATE, 
                                                                        start_epoch, reals_shown_now))
else:
    print("=> NO CHECKPOINT FOUND AT '{}'" .format(PATH_SAVE_NETWORK_INTERMEDIATE))
    raise RuntimeError("No checkpoint found at specified path.")

net.eval()

# Visualisation
# generate a batch sample from the dataloader
# saves images in the same directory as the trained network
sample = next(iter(dl))
print(sample["source"].shape)
out = net(sample["source"])
out = out.detach()
# Save directory for images
save_directory = os.path.split(PATH_SAVE_NETWORK_INTERMEDIATE)[0]
for batch_idx in range(_batch_size):
    if "boneless" in keys_images:
        plt.figure(1)
        fig, ax = plt.subplots(1,3, figsize=(15,5))
        ax[0].imshow(sample["source"][batch_idx,0,:],cmap='gray')
        ax[0].set_title("Source")
        ax[0].axis("off")
        ax[1].imshow(out[batch_idx,0,:],cmap='gray')
        ax[1].set_title("Suppressed")
        ax[1].axis("off")
        ax[2].imshow(sample["boneless"][batch_idx,0,:],cmap='gray')
        ax[2].set_title("Ideal")
        ax[2].axis("off")
    else:
        plt.figure(1)
        fig, ax = plt.subplots(1,2, figsize=(15,5))
        ax[0].imshow(sample["source"][batch_idx,0,:],cmap='gray')
        ax[0].set_title("Source")
        ax[0].axis("off")
        ax[1].imshow(out[batch_idx,0,:],cmap='gray')
        ax[1].set_title("Suppressed")
        ax[1].axis("off")
    if flag_savePictures:
        plt.savefig(os.path.join(save_directory, switch + "_comparisonImages_"+ str(batch_idx) +".png"))