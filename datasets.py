import torch
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os, sys, time, datetime, pathlib, random, math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tvtransforms
from skimage import io, transform
from PIL import Image


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
        if data_BSE is None:
            sample = {"Patient": [], "source":[]}
        else:
            sample = {"Patient": [], "boneless":[], "source":[]}
            
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
        source_image = source_image*255
        source_image = source_image.astype(np.uint8)
        source_image = Image.fromarray(source_image).convert("L")
            
        if "boneless" in self.data.keys():
            
            boneless_image = plt.imread(self.data["boneless"].iloc[idx])
            boneless_image = boneless_image*255
            boneless_image = boneless_image.astype(np.uint8)
            boneless_image = Image.fromarray(boneless_image).convert("L")
            
            #source_image, boneless_image = _check_if_array_3D(source_image, boneless_image)
            sample = {'source': source_image, 'boneless': boneless_image, 'Patient': patient_code}
        else:
            #source_image = _check_if_array_3D(source_image, None)
            sample = {'source': source_image, 'Patient': patient_code}
        
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
########################
# QEH Dataset
########################
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
        source_image = source_image*255
        source_image = source_image.astype(np.uint8)
        source_image = Image.fromarray(source_image).convert("L")
        #source_image = _check_if_array_3D(source_image)
        
        sample = {'source': source_image, 'Patient': patient_code}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
#########################
# Dongrong Dataset and similar dataset structures
#########################
class DongrongCOVIDDataset(Dataset):
    def __init__(self, normal_path, pneumonia_path, covid_path, transform=None):
        """
        Args:
            [normal, pneumonia, covid]_path: path to image directory containing [normal, pneumonia, covid] images.
            transform: optional transform to be applied on a sample.
        """
        image_names=[]
        labels=[]
        #normal_path= os.path.join(dataset_path,'NORMAL')
        #pneumonia_path= os.path.join(dataset_path,'PNEUMONIA')
        #covid_path=os.path.join(dataset_path,'COVID')
        
        if normal_path is not None:
            normalDir = os.listdir(normal_path)
            for allDir in normalDir:
                normal_image_name = os.path.join('%s\\%s' % (normal_path, allDir))
                image_names.append(normal_image_name)
                label=0
                labels.append(label)
        if pneumonia_path is not None:
            pneumoniaDir = os.listdir(pneumonia_path)
            for allDir in pneumoniaDir:
                pneumonia_image_name = os.path.join(pneumonia_path, allDir)
                image_names.append(pneumonia_image_name)
                label=1
                labels.append(label)
        
        if covid_path is not None:
            covidDir = os.listdir(covid_path)
            for allDir in covidDir:
                covid_image_name = os.path.join('%s\\%s' % (covid_path, allDir))
                image_names.append(covid_image_name)
                label = 2
                labels.append(label)
        
        package=list(zip(image_names,labels))
        random.shuffle(package)
        image_names[:],labels[:]=zip(*package)
        
        self.image_names = image_names
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        """
        Do not convert image to RGB for bone suppression
        Input:
            index: the index of item
        Output:
            sample: a dict with the fields:
                "source": the image
                "label": the label associated with the image
                "Patient": the name of the image file
        """
        image_name = self.image_names[index]
        patient_code = os.path.splitext(image_name)[0]
        image = Image.open(image_name) # do NOT convert to RGB for bone suppression
        
        # For Rajaraman, don't care about whether the image is normal, pneumonia or covid
        label = self.labels[index]
        sample = {'source': image, 'label':label, 'Patient': patient_code}
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample  #, torch.tensor(label)#torch.FloatTensor(label)

#########################
# Yuhua Dataset
#########################
class Yuhua_DDR(Dataset):
    def __init__(self, external_test_file, transform=None):
        self.transform = transform
        # Load data
        data = np.load(external_test_file) # [H x W x N]
        # insert channel dim
        data = np.expand_dims(data,-2) # [H x W x C x N]
        data = np.flip(data,0)
        self.data = data # numpy
    def __len__(self):
        return self.data.shape[-1]
    def __getitem__(self, index):
        image = self.data[:,:,:,index].copy()
        sample = {'source': image, "Patient":index}
        if self.transform is not None:
            sample = self.transform(sample)
        else:
            sample['source'] = tvtransforms.ToTensor(image)
        return sample