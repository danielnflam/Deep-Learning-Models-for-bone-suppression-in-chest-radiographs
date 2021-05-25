import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys, time, datetime, pathlib, random, math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tvtransforms
from skimage import io, transform

# HELPER FUNCTION
def _check_if_array_3D(source_image, boneless_image=None):
    # Check if array is 3D or 2D
    iters = 0
    img_list = [source_image, boneless_image]
    for image in img_list:
        if image is not None:
            if image.ndim == 3:
                # make the image grayscale
                image = image[:,:,0]
            iters+=1
            if iters == 1:
                source_image = image
            if iters == 2:
                boneless_image = image
    
    if boneless_image is None:
        return source_image
    else:
        return source_image, boneless_image

###########################
# JSRT CXR dataset
# Shiraishi J, Katsuragawa S, Ikezoe J, Matsumoto T, Kobayashi T, Komatsu K, Matsui M, Fujita H, Kodera Y, and Doi K.: Development of a digital image database for chest radiographs with and without a lung nodule: Receiver operating characteristic analysis of radiologists’ detection of pulmonary nodules. AJR 174; 71-74, 2000
###########################
class JSRT_CXR(Dataset):
    def __init__(self, data_normal, data_BSE, transform):
        """
        Inputs:
            data_normal: root directory holding the normal / non-suppressed images
            data_BSE: root directory holding the bone-suppressed images
            transform: (optional) a torchvision.transforms.Compose series of transformations
        Assumed that files corresponding to the same patient have the same name in both folders data_normal and data_BSE.
        """
        if data_BSE is not None:
            sample = {"Patient": [], "boneless":[], "source":[]}
        else:
            sample = {"Patient": [], "source":[]}
        for root, dirs, files in os.walk(data_normal):
            for name in files:
                if '.png' in name:
                    a_filepath = os.path.join(root, name)
                    # Patient code
                    head, tail = os.path.split(a_filepath)
                    patient_code_file = os.path.splitext(tail)[0]
                    # Place into lists
                    sample["Patient"].append(patient_code_file)
                    sample["source"].append(a_filepath)
                    
                    # For each patient code, search the alternate data_folder to obtain the corresponding source
                    if data_BSE is not None:
                        for root2, dirs2, files2 in os.walk(data_BSE):
                            for name2 in files2:
                                # Need regex to distinguish between e.g. 0_1 and 0_10
                                filename2,_ = os.path.splitext(name2)
                                if patient_code_file == filename2:
                                    sample["boneless"].append(os.path.join(root2, name2))

        self.data = pd.DataFrame(sample)        
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Describe the reading of images in here"""
        if torch.is_tensor(idx):
            idx = idx.tolist() # transform into python list
        
        patient_code = self.data["Patient"].iloc[idx]
        source_image = plt.imread(self.data["source"].iloc[idx])
        boneless_image = plt.imread(self.data["boneless"].iloc[idx])
        source_image, boneless_image = _check_if_array_3D(source_image, boneless_image)
        
        sample = {'source': source_image, 'boneless': boneless_image} #'patientCode': patient_code
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
        
    def visualise(self, idx):
        bonelessIm = plt.imread(self.data["boneless"].iloc[idx])
        sourceIm = plt.imread(self.data["source"].iloc[idx])
        sourceIm, bonelessIm = _check_if_array_3D( sourceIm, bonelessIm)
        
        # Visualisation
        fig, ax=plt.subplots(1,2)
        ax[0].imshow(sourceIm, cmap="gray")
        ax[1].imshow(bonelessIm, cmap="gray")
    


class POLYU_COVID19_CXR_CT_Cohort1(Dataset):
    def __init__(self, data_normal, transform):
        """
        Inputs:
            data_normal: root directory holding the normal / non-suppressed images
            transform: (optional) a torchvision.transforms.Compose series of transformations
        Assumed that files corresponding to the same patient have the same name in both folders data_normal and data_BSE.
        """
        sample = {"Patient": [], "source":[]}
        for root, dirs, files in os.walk(data_normal):
            for name in files:
                if '.png' in name:
                    a_filepath = os.path.join(root, name)
                    # Patient code
                    head, tail = os.path.split(a_filepath)
                    patient_code_file = os.path.splitext(tail)[0]
                    # Place into lists
                    sample["Patient"].append(patient_code_file)
                    sample["source"].append(a_filepath)

        self.data = pd.DataFrame(sample)
        self.transform = transform
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Describe the reading of images in here"""
        if torch.is_tensor(idx):
            idx = idx.tolist() # transform into python list
        
        patient_code = self.data["Patient"].iloc[idx]
        source_image = plt.imread(self.data["source"].iloc[idx])
        source_image = _check_if_array_3D(source_image)
        
        sample = {'source': source_image} #'patientCode': patient_code
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample