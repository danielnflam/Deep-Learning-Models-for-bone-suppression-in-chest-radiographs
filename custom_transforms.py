import torch
import pandas as pd
import numpy as np
#from skimage import io, transform 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import os, sys, time, datetime
import skimage.exposure
from PIL import Image, ImageOps

"""
File contains transformations as callable classes.
PIL Image and Torch Tensors are the assumed inputs for all functions here.
"""
###############
# PIL Images
###############

class ToTensor(object):
    def __init__(self, sample_keys_images):
        self.sample_keys_images = sample_keys_images
        self.tform = transforms.ToTensor()
    def __call__(self, sample):
        for key_idx in self.sample_keys_images:
            image = sample[key_idx]
            out = self.tform(image)
            sample[key_idx] = out
        return sample

class HistogramEqualisation(object):
    """
    Works for PIL images only.
    """
    def __init__(self, sample_keys_images, mask=None):
        self.sample_keys_images = sample_keys_images
        self.mask = mask
    def __call__(self, sample):
        for key_idx in self.sample_keys_images:
            image = sample[key_idx]
            image = ImageOps.equalize(image, self.mask)
            sample[key_idx] = image
        return sample

class Resize(object):
    """
    Resize the image in a sample to a given size.
    This effectively resamples the image to fit that given output size.    
    Input integer decides the number of column pixels.    
    To keep the aspect ratio the same, use an integer as the input when initialising this object.
    """
    def __init__(self, sample_keys_images, output_size):
        """
        Inputs:
            output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, that will be the number of column pixels, 
            and the number of row pixels is determined from the aspect ratio
            sample_keys_images (list or tuple): list of strings representing the keys to images in the sample_dictionary
        """
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.sample_keys_images = sample_keys_images
        self.tform = transforms.Resize(self.output_size)
    def __call__(self, sample):
        """
        Inputs:
            sample (dict): the dictionary containing the images to be transformed
                            Images should be PIL Images or pytorch Tensors.
        """
        for key_idx in self.sample_keys_images:
            image = sample[key_idx]
            out = self.tform(image)
            sample[key_idx] = out
        return sample
    
class CenterCrop(object):
    def __init__(self, sample_keys_images, output_size):
        self.output_size = output_size
        self.sample_keys_images = sample_keys_images
        self.tform = transforms.CenterCrop(self.output_size)
    def __call__(self, sample):
        for key_idx in self.sample_keys_images:
            image = sample[key_idx]
            out = self.tform(image)
            sample[key_idx] = out
        return sample

class RandomHorizontalFlip(object):
    def __init__(self, sample_keys_images, probability=0.5):
        self.probability = probability
        self.sample_keys_images = sample_keys_images
        self.tform = transforms.RandomHorizontalFlip(self.probability)
    def __call__(self, sample):
        for key_idx in self.sample_keys_images:
            image = sample[key_idx]
            out = self.tform(image)
            sample[key_idx] = out
        return sample

class RandomAffine(object):
    """
    Input is a torch tensor.
    """
    def __init__(self, sample_keys_images, degrees=10,translate=(0.1,0.1),scale=(0.9,1.1), shear=(0,0)):
        self.sample_keys_images =sample_keys_images
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
    def transform(self, normal, suppressed):
        angle = np.random.uniform(-self.degrees, self.degrees)
        translates = [np.random.uniform(-self.translate[0],self.translate[0]) , np.random.uniform(-self.translate[1],self.translate[1]) ]
        scale = np.random.uniform(min(self.scale),max(self.scale))
        shear = np.random.uniform(min(self.shear),max(self.shear))
        
        # Identify mean intensity in diaphragm area
        width, height = normal.size
        min_intensity, max_intensity = normal.getextrema()
        
        lowest10percent = round(height*0.9)
        average_intensity_threshold = (max_intensity - min_intensity)//2 + min_intensity
        test = np.array(normal.copy())
        # This following line kills the images
        average_lowest10percent_intensity = np.mean(test[lowest10percent:height,:])
        
        if average_lowest10percent_intensity > average_intensity_threshold:
            # i.e. attenuated regions are white; fill value is black
            fill_value = min_intensity
        else:
            fill_value = max_intensity
        
        # Fillcolor doesn't work
        normal = TF.affine(normal , angle, translates, scale, shear, fill=fill_value)
        if suppressed is not None:
            suppressed = TF.affine(suppressed , angle, translates, scale, shear, fill=fill_value)
        return normal, suppressed
    
    def __call__(self, sample):
        image = sample[self.sample_keys_images[0]]
        suppressed = sample[self.sample_keys_images[1]]
        
        image, suppressed = self.transform(image, suppressed)
        
        sample[self.sample_keys_images[0]] = image
        sample[self.sample_keys_images[1]] = suppressed
        return sample
        
        
####################
# TORCH TENSORS ONLY
####################
class Normalize_ZeroToOne(object):
    def __init__(self, sample_keys_images):
        self.sample_keys_images = sample_keys_images
    def __call__(self, sample):
        for key_idx in self.sample_keys_images:
            image = sample[key_idx]
            max_image = torch.max(image)
            min_image = torch.min(image)
            out = (image-min_image)/(max_image-min_image)
            sample[key_idx] = out
        return sample

class ImageComplement(object):
    # Flip image intensities (i.e. black becomes white, white becomes black)
    def __init__(self, sample_keys_images, probability=0.5):
        self.probability = probability
        self.sample_keys_images = sample_keys_images
    def __call__(self, sample):
        activate = np.random.uniform(0,1)
        for key_idx in self.sample_keys_images:
            image = sample[key_idx]
            
            if activate < self.probability:
                max_image = torch.max(image)
                min_image = torch.min(image)
                image = (image-min_image)/(max_image-min_image) # range [0,1]
                image = (1-image)*(max_image-min_image) + min_image
            
            sample[key_idx] = image
        return sample
    

##########################################
