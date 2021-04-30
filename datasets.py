import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys, time, datetime, pathlib, random, math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tvtransforms
from skimage import io, transform

###########################
# JSRT CXR dataset
# Shiraishi J, Katsuragawa S, Ikezoe J, Matsumoto T, Kobayashi T, Komatsu K, Matsui M, Fujita H, Kodera Y, and Doi K.: Development of a digital image database for chest radiographs with and without a lung nodule: Receiver operating characteristic analysis of radiologists’ detection of pulmonary nodules. AJR 174; 71-74, 2000
###########################
class JSRT_CXR(Dataset):
    def __init__(self, data_normal, data_BSE):
        sample = {"Patient": [], "boneless":[], "source":[]}
        for root, dirs, files in os.walk(data_BSE):
            for name in files:
                if '.png' in name:
                    a_filepath = os.path.join(root, name)
                    # Patient code
                    head, tail = os.path.split(a_filepath)
                    patient_code_file = os.path.splitext(tail)[0]
                    # Place into lists
                    sample["Patient"].append(patient_code_file)
                    sample["boneless"].append(a_filepath)

                    # For each patient code, search the alternate data_folder to obtain the corresponding source
                    for root2, dirs2, files2 in os.walk(data_normal):
                        for name2 in files2:
                            if patient_code_file in name2:
                                sample["source"].append(os.path.join(root2, name2))
        self.data = pd.DataFrame(sample)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Describe the reading of images in here"""
        if torch.is_tensor(idx):
            idx = idx.tolist() # transform into python list
        
        source_image = plt.imread(self.data["source"].iloc[idx])
        boneless_image = plt.imread(self.data["boneless"].iloc[idx])
        source_image, boneless_image = self._check_if_array_3D(source_image, boneless_image)
        
        patient_code = self.data["Patient"].iloc[idx]
        
        sample = {'source': source_image, 'boneless': boneless_image, 'patientCode': patient_code}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
        
    def visualise(self, idx):
        bonelessIm = plt.imread(self.data["boneless"].iloc[idx])
        sourceIm = plt.imread(self.data["source"].iloc[idx])
        sourceIm, bonelessIm = self._check_if_array_3D( sourceIm, bonelessIm)
        
        # Visualisation
        fig, ax=plt.subplots(1,2)
        ax[0].imshow(sourceIm, cmap="gray")
        ax[1].imshow(bonelessIm, cmap="gray")
    
    # Helper function
    def _check_if_array_3D(self, source_image, boneless_image):
        # Check if array is 3D or 2D
        iters = 0
        for image in [source_image, boneless_image]:
            if image.ndim == 3:
                # make the image grayscale
                image = image[:,:,0]
            iters+=1
            if iters == 1:
                source_image = image
            if iters == 2:
                boneless_image = image
        return source_image, boneless_image