import torch
import pandas as pd
import numpy as np
from skimage import io, transform 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os, sys, time, datetime
import skimage.exposure

"""
File contains transformations as callable classes.

"""



class ToTensor(object):
    """
    Torch transforms generally work on Torch tensor datasets best.
    This transform will turn a numpy ndarray into a torch tensor.
    
    No init -- this is a standard function.
    """
    def __init__(self, sample_keys):
        self.sample_keys_images = sample_keys
    def __call__(self, sample):
        """
        Input:
            sample (dict): the dictionary containing the images to be transformed
        """
        for key_idx in self.sample_keys_images:
            image = sample[key_idx]
            output = self.execute(image)
            sample[key_idx] = output
        return sample
    def execute(self, image):
        image = image.astype(np.float32)
        if image.ndim==2:
            output = torch.from_numpy(image[np.newaxis,:])
        elif image.ndim==3:
            output = torch.from_numpy(image)
        return output
    
class HistogramEqualisation(object):
    def __init__(self, sample_keys_images):
        self.sample_keys_images = sample_keys_images
    def __call__(self, sample):
        for key_idx in self.sample_keys_images:
            image = sample[key_idx]
            image = skimage.exposure.equalize_hist(image)
            sample[key_idx] = image
        return sample

class CLAHE(object):
    def __init__(self, sample_keys_images):
        self.sample_keys_images = sample_keys_images
    def __call__(self, sample):
        for key_idx in self.sample_keys_images:
            image = sample[key_idx]
            image = skimage.exposure.equalize_adapthist(image)
            sample[key_idx] = image
        return sample
            

class ZScoreNormalisation(object):
    def __init__(self, sample_keys_images):
        self.sample_keys_images = sample_keys_images
    def __call__(self, sample):
        for key_idx in self.sample_keys_images:
            image = sample[key_idx]
            image = self.execute(image)
            sample[key_idx] = image
        return sample
    def execute(self, image):
        image = (image - np.mean(image))/np.std(image)
        return image

class NormaliseBetweenPlusMinus1(object):
    def __init__(self, sample_keys_images):
        self.sample_keys_images = sample_keys_images
    def __call__(self, sample):
        for key_idx in self.sample_keys_images:
            image = sample[key_idx]
            new_image = self.execute(image)
            sample[key_idx] = new_image
        return sample
    def execute(self, image):
        # Rescale to between 0 and 1
        new_image = (image - np.amin(image)) / (np.amax(image) - np.amin(image))
        # Rescale to between -1 and 1
        new_image = (new_image*2 - 1)
        return new_image

class RandomIntensityComplement(object):
    # Black becomes white and white becomes black
    def __init__(self, sample_keys_images, probability=0.5):
        self.probability = probability
        self.sample_keys_images = sample_keys_images
        
    def __call__(self, sample):
        if np.random.rand(1) < self.probability:
            for key_idx in self.sample_keys_images:
                
                image = sample[key_idx]
                max_image = np.amax(image)
                min_image = np.amin(image)
                # Rescale image to [0,1]
                image = (image - min_image)/(max_image - min_image)
                # Flip image
                flipped_image = 1 - image
                # Restore previous scale
                flipped_image = flipped_image*(max_image - min_image) + min_image
                sample[key_idx] = flipped_image
                
        return sample
    
class IntensityJitter(object):
    """
    Scale the intensity of the input image randomly by a factor that is randomly chosen between the rescale_factor_limits.
    
    Images input are numpy NDARRAYS
    """
    def __init__(self, sample_keys_images, source_image_key="source", rescale_factor_limits=(0.75,1.0), window_motion_limits=(-1,1)):
        self.sample_keys_images = sample_keys_images
        self.rescale_factor_limits = rescale_factor_limits
        self.window_motion_limits = window_motion_limits
        self.source_image_key = source_image_key
    def __call__(self, sample):
        # Generate the same factor for all images denoted by the sample_keys
        
        sd_image = np.std(sample[self.source_image_key])
        mean_image = np.mean(sample[self.source_image_key])
        
        window_motion = np.random.rand(1)*(max(self.window_motion_limits) - min(self.window_motion_limits)) + min(self.window_motion_limits)
        intensity_factor = np.random.rand(1)*(max(self.rescale_factor_limits) - min(self.rescale_factor_limits)) + min(self.rescale_factor_limits)
        
        for key_idx in self.sample_keys_images:
            image = sample[key_idx]
            # Z-Normalise
            standardised_image = (image-mean_image)/sd_image
            # Change intensity distributions
            standardised_image = standardised_image*intensity_factor
            # Replace image
            sample[key_idx] = standardised_image*sd_image + mean_image + window_motion*sd_image
        return sample

class Rescale(object):
    """
    Rescale the image in a sample to a given size.
    This effectively resamples the image to fit that given output size.
    
    Input integer decides the number of column pixels.
    
    To keep the aspect ratio the same, use an integer as the input when initialising this object.
    """
    def __init__(self, output_size, sample_keys_images, sample_key_PixelSize=None):
        """
        Inputs:
            output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, that will be the number of column pixels, 
            and the number of row pixels is determined from the aspect ratio
            sample_keys_images (list or tuple): list of strings representing the keys to images in the sample_dictionary
            sample_key_PixelSize (string): string for the key holding the PixelSize values in the sample dict.
        """
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.sample_keys_images = sample_keys_images
        #self.sample_key_PixelSize = sample_key_PixelSize
        
    def __call__(self, sample):
        """
        Inputs:
            sample (dict): the dictionary containing the images to be transformed
                            Images should be numpy arrays in the format: [H x W]
                            B is batch number, C is channels, H is number of rows, W is number of columns
                            The "ToTensor" function can do this.
        """
        for key_idx in self.sample_keys_images:
            image = sample[key_idx]
            #if self.sample_key_PixelSize is not None:
            #    PixelSize = sample[self.sample_key_PixelSize]
            #else:
            #    PixelSize = None
            
            if len(image.shape) == 2:
                h, w = image.shape[:2]
                
            if isinstance(self.output_size, int):
                new_h, new_w = self.output_size* h / w, self.output_size
            else:
                new_h, new_w = self.output_size

            new_h, new_w = int(new_h), int(new_w)

            if len(image.shape)==2:
                # Skimage transform
                out = transform.resize(image, (new_h, new_w), order=0)
            
            sample[key_idx] = out
            
            # Output the rescaled PixelSize
            #if PixelSize == None:
            #    out_PixelSize = None
            #else:
            #    out_PixelSize = (PixelSize[0]*(h/new_h), PixelSize[1]*(w/new_w))
            #sample[self.sample_key_PixelSize] = out_PixelSize
        return sample

class RescalingNormalisation(object):
    def __init__(self, sample_keys_images, rescale_range):
        """
        Inputs:
            sample_keys_images (list or tuple): list of strings representing the keys to images in the sample_dictionary
        """
        self.sample_keys_images = sample_keys_images
        self.rescale_range = rescale_range
    def __call__(self, sample):
        """
        Inputs:
            sample (dict): the dictionary containing the images to be transformed
                            Images should be numpy arrays in the format [H x W]
                            B is batch number, C is channels, H is number of rows, W is number of columns
                            The "ToTensor" function can do this.
        """
        for key_idx in self.sample_keys_images:
            image = sample[key_idx]
            output = self.execute(image)
            sample[key_idx] = output
        return sample
    def execute(self, image):
        scaled_image = (image - np.amin(image))/(np.amax(image) - np.amin(image)) # range[0,1]
        output = scaled_image*(max(self.rescale_range) - min(self.rescale_range)) + min(self.rescale_range) # range[min(self.rescale_range), max(self.rescale_range)]
        return output
        
class ImageComplement(object):
    """
    Inverse the grayscale image.  White becomes black and vice versa.
    """
    def __init__(self, sample_keys_images):
        self.sample_keys_images = sample_keys_images
    def __call__(self, sample):
        """
        Image in sample_dict assumed to be a numpy array [H x W]
        """
        for key_idx in self.sample_keys_images:
            image = sample[key_idx]
            flipped_image = self.execute(image)
            sample[key_idx] = flipped_image
        return sample
    def execute(self, image):
        max_image = np.amax(image)
        min_image = np.amin(image)
        # Rescale image to [0,1]
        image = (image - min_image)/(max_image - min_image)
        # Flip image
        flipped_image = 1 - image
        # Restore previous scale
        flipped_image = flipped_image*(max_image - min_image) + min_image
        return flipped_image
    
class Random180(object):
    """
    Randomly flip image via horizontal axis and then vertical axis.
    This ensures that the heart points towards the left side of the body.
    Probability: the % chance that the flip occurs.  Probability = 0.6 means that a flip occurs 60% of the time.
    """
    def __init__(self, sample_keys_images, probability=0.5):
        self.sample_keys_images = sample_keys_images
        self.probability = probability
    def __call__(self, sample):
        if np.random.rand(1) < self.probability:
            for key_idx in self.sample_keys_images:
                image = sample[key_idx]
                image = np.flip(image, 0)
                image = np.flip(image, 1)
                sample[key_idx] = image
        
        return sample
    
class RandomHorizontalFlip(object):
    """
    Randomly flip image via horizontal axis.
    Probability: the % chance that the flip occurs.  Probability = 0.6 means that a flip occurs 60% of the time.
    """
    def __init__(self, sample_keys_images, probability=0.5):
        self.sample_keys_images = sample_keys_images
        self.probability = probability
    def __call__(self, sample):
        if np.random.rand(1) < self.probability:
            for key_idx in self.sample_keys_images:
                image = sample[key_idx]
                image = np.flip(image, 0)
                sample[key_idx] = image
        return sample

class RandomVerticalFlip(object):
    """
    Randomly flip image via horizontal axis.
    Probability: the % chance that the flip occurs.  Probability = 0.6 means that a flip occurs 60% of the time.
    """
    def __init__(self, sample_keys_images, probability=0.5):
        self.sample_keys_images = sample_keys_images
        self.probability = probability
    def __call__(self, sample):
        if np.random.rand(1) < self.probability:
            for key_idx in self.sample_keys_images:
                image = sample[key_idx]
                image = np.flip(image, 1)
                sample[key_idx] = image
        return sample
    
class RandomRotation(object):
    def __init__(self, sample_keys_images):
        self.sample_keys_images = sample_keys_images
        
    def __call__(self, sample):
        angle = np.random.rand(1)*360
        for key_idx in self.sample_keys_images:
            image = sample[key_idx]
            image = transform.rotate(image, angle, resize=False)
            sample[key_idx] = image
        return sample