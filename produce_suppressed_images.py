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

# Paths
PATH_SAVE_NETWORK_INTERMEDIATE = "./trained_network.tar"
key_source = "source" # this is the dictionary key for the original radiograph in the datasets
key_boneless = "boneless" # this is the dictionary key for the bone-suppressed radiograph in the datasets file
# Data
_batch_size = 10
image_spatial_size = (440,440)

switch = "external_POLYU" #"internal"##

print("The dataset chosen is: " + switch)
if switch == "internal":
    directory_source = "D:/data/JSRT/augmented/test/source/"
    directory_boneless = "D:/data/JSRT/augmented/test/target/"
    keys_images = [key_source, key_boneless]
    ds = datasets.JSRT_CXR(directory_source, directory_boneless, 
                           transform=tvtransforms.Compose([
                                 transforms.RescalingNormalisation(keys_images,(0,1)),
                                 transforms.Rescale(image_spatial_size, keys_images, None),
                                 transforms.ToTensor(keys_images),
                                 ]))
elif switch == "external_POLYU":
    externalTest_directory = "D:/data/POLYU_COVID19_CXR_CT_Cohort1/cxr/CXR_PNG"
    keys_images = [key_source]
    ds = datasets.POLYU_COVID19_CXR_CT_Cohort1(externalTest_directory,
                                 transform=tvtransforms.Compose([
                                 transforms.RescalingNormalisation(keys_images,(0,1)),
                                 transforms.Rescale(image_spatial_size, keys_images, None),
                                 transforms.ToTensor(keys_images),
                                 ]))
else:
    raise RuntimeError("Dataset unknown.  Please input the details in the datasets.py file")
print(len(ds))
dl = DataLoader(ds, _batch_size, shuffle=True, num_workers=0)

## Code for putting things on the GPU
ngpu = 1 #torch.cuda.device_count()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)
if (torch.cuda.is_available()):
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    

## Network
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

net = net.to(device)
# Set to testing mode
net.eval()

## SAVE THE BONE SUPPRESSED IMAGES AS PNGs
path_to_save_images = Path(os.path.join("bone_suppressed",switch))
path_to_save_images.mkdir(parents=True, exist_ok=True)
iters=0
for ii, data in enumerate(dl):
    input_data = data[key_source].to(device)
    out = net(input_data)
    print(out.shape)
    print("Batch Number:" + str(ii))
    out = out.cpu()
    for image in out:
        savename = str(iters)+".png"
        vutils.save_image( image, os.path.join(path_to_save_images, savename))
        iters+=1
    