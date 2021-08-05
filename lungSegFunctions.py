import os, sys, math, datetime, random, copy
import numpy as np
import torch

# Lung segmentation networks
from lungVAE.models import VAE
from lungVAE.utils.postProcess import postProcess
from skimage.transform import resize
from skimage.exposure import equalize_hist as equalize
import torchvision.transforms.functional as TF
import torchvision.transforms as tvtransforms

import custom_transforms

# Lung Network Class
import lungVAE.models.VAE as VAE

class LungSegmentationNetwork():
    def __init__(self, device):
        
        # DEFAULT SETTINGS
        model="./lungVAE/saved_models/lungVAE.pt"
        hidden=16
        latent=8
        unet=False
        
        self.no_post = False
        self.p = 32 # padding
        self.no_preprocess = True
        
        print("Loading "+model)
        if 'unet' in model:
            unet = True
            hidden = int(1.5*hidden)
        else:
            unet = False
            
        # Load network
        net_lungSeg = VAE.uVAE(nhid=hidden,nlatent=latent,unet=unet)
        net_lungSeg.load_state_dict(torch.load(model, map_location=device))
        net_lungSeg.to(device)
        nParam = sum(p.numel() for p in net_lungSeg.parameters() if p.requires_grad)
        print("Model "+model.split('/')[-1]+" Number of parameters:%d"%(nParam))
        
        self.network = net_lungSeg
        self.device = device
    
    def segment(self, image_minibatch):
        # Image is [NxCxHxW]
        # Original image is equalised if no_preprocess=False
        image_minibatch = image_minibatch.to(self.device)
        output_data = lungSegmentation_maskOnly(self.network, image_minibatch, self.device, 
                                                p=self.p, no_preprocess=self.no_preprocess, 
                                                standardisedMonochrome="MONOCHROME1",no_post=self.no_post)
        # Outputs
        mask = output_data["mask"]
        image = output_data["image"]
        return mask, image
    
    def crop(self, image_minibatch, mask_minibatch, image_spatial_size=None, interp_mode=TF.InterpolationMode.NEAREST):
        # image_minibatch is [NxCxHxW]
        # mask_minibatch is [NxCxHxW]
        if image_spatial_size is not None:
            torchresize = tvtransforms.Resize(image_spatial_size, interpolation=interp_mode)
        # Find BB using mask
        outputs = []
        for idx2, maskCHW in enumerate(mask_minibatch):
            bb = BoundingBox(maskCHW)
            indices = bb.findBox()
            croppedImage = image_minibatch[idx2,:,indices["topbottom"][0]:indices["topbottom"][1]+1, indices["leftright"][0]:indices["leftright"][1]+1]
            if image_spatial_size is not None:
                croppedImage = torchresize(croppedImage)
                outputs.append(croppedImage)
            else:
                outputs.append(croppedImage)
        outputs = torch.stack(outputs)
        return outputs
            

# Function to load data
def preProcessing(dcm, p=32, no_preprocess=False, standardisedMonochrome="MONOCHROME1", verbose=False):
    # input expected to be torch tensor [ ... , H, W]
    # output is [1x1xHxW]
    wLoc = 448
    imH = dcm.shape[-2]
    imW = dcm.shape[-1]
    ## Preprocessing
    if not no_preprocess:
        # torch tensor to numpy then back to torch tensor
        print("Equalising")
        dcm = dcm.squeeze().numpy()
        dcm = equalize(dcm)
        dcm = torch.Tensor(dcm).unsqueeze(0) # [1xHxW]
    else:
        dcm = dcm.squeeze()
        dcm = dcm.unsqueeze(0) # [1xHxW]
    
    ### Crop and resize image to 640x512 
    hLoc = int((dcm.shape[-2]/(dcm.shape[-1]/wLoc)))
    if hLoc > 576:
        hLoc = 576
        wLoc = int((dcm.shape[-1]/(dcm.shape[-2]/hLoc)))
    
    img = TF.resize(dcm,(hLoc,wLoc))
    if standardisedMonochrome is not None:
        standardiseMonochrome = custom_transforms.StandardiseMonochrome(None, standardisedMonochrome, verbose)
        img , has_switched = standardiseMonochrome.tform(img)
        #print("Has switched?: {}".format(has_switched))
    
    pImg = torch.zeros((1, 640,512))
    h = (int((576-hLoc)/2))+p
    w = int((448-wLoc)/2)+p
    roi = torch.zeros(pImg.shape).squeeze()
    
    if w == p:
        height = img.shape[-2]
        pImg[:,np.abs(h):(h+height),p:-p] = img
        roi[np.abs(h):(h+height),p:-p] = 1.0
    else:
        width = img.shape[-1]
        pImg[:,p:-p,np.abs(w):(w+width)] = img
        roi[p:-p,np.abs(w):(w+width)] = 1.0
    pImg=pImg.unsqueeze(0)
    return pImg, roi, h, w, hLoc, wLoc, imH, imW

def postProcessMask(img,h,w,hLoc,wLoc,imH,imW,no_post=False,p=32):
    # img is a torch tensor
    img = img.detach()
    imgIp = img.detach().clone()
    if w == p:
        img = TF.resize(img[:,:,np.abs(h):(h+hLoc),p:-p],
                    (imH,imW))
    else:
        img = TF.resize(img[:,:,p:-p,np.abs(w):(w+wLoc)],
                    (imH,imW))
    
    if not no_post:
        imgIp = imgIp.squeeze().data.numpy()
        imgPost = postProcess(imgIp)
        imgPost = torch.from_numpy(imgPost).unsqueeze(0).unsqueeze(0)
        
        if w == p:
            imgPost = TF.resize(imgPost[:,:,np.abs(h):(h+hLoc),p:-p],
                            (imH,imW), interpolation=TF.InterpolationMode.NEAREST)
        else:
            imgPost = TF.resize(imgPost[:,:,p:-p,np.abs(w):(w+wLoc)],
                            (imH,imW), interpolation=TF.InterpolationMode.NEAREST)
    else:
        imgPost = img
    return imgPost > 0.5

def lungSegmentation_maskOnly(net_lungSeg, input_minibatch, device, p = 32, no_preprocess=False, standardisedMonochrome="MONOCHROME1",no_post=False):
    # input_minibatch is [NxCxHxW]
    data = {"mask":[], "image":[]}
    for image in input_minibatch:
        image = image.unsqueeze(0) #[1x1xHxW]
        if not no_preprocess:
            image = image.squeeze().numpy() #[HxW]
            image = equalize(image)
            image = torch.Tensor(image).unsqueeze(0).unsqueeze(0) #[1x1xHxW]
        data["image"].append(image) # original image
        
        # preProcessing of image is already handled above
        img, roi, h, w, hLoc, wLoc, imH, imW = preProcessing(image, p, no_preprocess=True,
                                               standardisedMonochrome=standardisedMonochrome)
        # Segment
        img = img.to(device)
        roi = roi.to(device)
        _,mask = net_lungSeg(img)
        mask = torch.sigmoid(mask*roi)
        mask = mask.cpu()
        
        mask = postProcessMask(mask,h,w,hLoc,wLoc,imH,imW,no_post=no_post,p=p) # resize to original image dimensions
        data["mask"].append(mask)
        
        
    data["mask"] = torch.cat(data["mask"],0)
    data["image"] = torch.cat(data["image"],0)
    return data



def lungSegmentation_RenGe(net_lungSeg, input_minibatch, boneless_image_minibatch=None, p=32, original_image_no_preprocess=False,
                           lung_seg_standardisedMonochrome="MONOCHROME1",
                           original_image_standardisedMonochrome="MONOCHROME2", no_post=False, verbose=False):
    
    # input_minibatch is [NxCxHxW]
    data = {"mask":[], "image":[], "boneless":[],
            "croppedImage":[], "croppedMask":[], "croppedBoneless":[]}
    output_data = copy.deepcopy(data)
    for idx, image in enumerate(input_minibatch):
        original_image = image.unsqueeze(0) # [1x1xHxW]
        # image to be segmented
        img, roi, h, w, hLoc, wLoc, imH, imW = preProcessing(original_image, p, no_preprocess=False,
                                                           standardisedMonochrome=lung_seg_standardisedMonochrome, verbose=verbose)
        # original image to be resized
        original_image, roi, h, w, hLoc, wLoc, imH, imW = preProcessing(original_image, p, no_preprocess=original_image_no_preprocess,
                                                                        standardisedMonochrome=original_image_standardisedMonochrome, verbose=verbose)
        if boneless_image_minibatch is not None:
            boneless_image, _, _, _, _, _, _, _ = preProcessing(boneless_image_minibatch[idx,:], p, no_preprocess=original_image_no_preprocess,
                                                                        standardisedMonochrome=original_image_standardisedMonochrome, verbose=verbose)
        
        #img = img.to(device)
        _,mask = net_lungSeg(img)
        mask = torch.sigmoid(mask*roi)
        
        # mask is post-processed
        if not no_post:
            maskIp = mask.squeeze().data.numpy()
            maskPost = postProcess(maskIp)
            maskPost = torch.from_numpy(maskPost).unsqueeze(0).unsqueeze(0)
        
        data["mask"].append(maskPost)
        data["image"].append(img) # original image
        if boneless_image_minibatch is not None:
            data["boneless"].append(boneless_image)
            
        # bounding box is identified
        bb = BoundingBox(maskPost)
        indices = bb.findBox()
        # crop the mask & original image
        croppedImage =original_image[:,:,indices["topbottom"][0]:indices["topbottom"][1]+1, indices["leftright"][0]:indices["leftright"][1]+1]
        croppedMask = maskPost[:,:,indices["topbottom"][0]:indices["topbottom"][1]+1, indices["leftright"][0]:indices["leftright"][1]+1]
        if boneless_image_minibatch is not None:
            croppedBoneless = boneless_image[:,:,indices["topbottom"][0]:indices["topbottom"][1]+1, indices["leftright"][0]:indices["leftright"][1]+1]
            
        # Reshape croppedImage and croppedMask to 256
        torchresize = tvtransforms.Resize((256,256))
        croppedImage = torchresize(croppedImage)
        croppedMask = torchresize(croppedMask)
        data["croppedImage"].append(croppedImage)
        data["croppedMask"].append(croppedMask)
        if boneless_image_minibatch is not None:
            croppedBoneless = torchresize(croppedBoneless)
            data["croppedBoneless"].append(croppedBoneless)
        
    
    output_data["mask"] = torch.cat(data["mask"],0)
    output_data["image"] = torch.cat(data["image"],0)
    output_data["croppedImage"] = torch.cat(data["croppedImage"],0)
    output_data["croppedMask"] = torch.cat(data["croppedMask"],0)
    
    if boneless_image_minibatch is not None:
        output_data["boneless"] = torch.cat(data["boneless"],0)
        output_data["croppedBoneless"] = torch.cat(data["croppedBoneless"],0)
        
    return output_data

def GaryPreprocessingInputData(data, keys_images, key_source, device, flag_segmentLung=False, net_lungSeg=None, p=32, no_post=False, 
                               flag_equaliseOriginalImages=False, flag_normalise=True, flag_cropping=True):
    
    # Ren Ge's network is trained on black-bone images
    # Standardise Input Image Monochrome
    data = standardiseMonochrome(data, keys_images, standard="MONOCHROME2", verbose=False)

    if flag_segmentLung:
        # Segment the lung mask
        # Original image is equalised if no_preprocess=False
        no_preprocess = not flag_equaliseOriginalImages # because equalisation will occur in the above section with data[key]
        output_data = LF.lungSegmentation_maskOnly(net_lungSeg, data[key_source].to(device), p=p, 
                                                no_preprocess=no_preprocess, standardisedMonochrome="MONOCHROME1",no_post=no_post)
    else:
        output_data = {'image':data[key_source]}

    # For boneless
    if key_boneless in data.keys():
        print("There are boneless images in the data.")
        output_data[key_boneless] = []
        if flag_equaliseOriginalImages:
            for image in data[key_boneless]:
                image = image.squeeze().numpy() #[HxW]
                image = skimage.exposure.equalize_hist(image)
                output_data[key_boneless].append(torch.Tensor(image).unsqueeze(0).unsqueeze(0)) #[1x1xHxW]
        else:
            for image in data[key_boneless]:
                image = image.squeeze().numpy() #[HxW]
                output_data[key_boneless].append(torch.Tensor(image).unsqueeze(0).unsqueeze(0)) #[1x1xHxW]
        output_data[key_boneless] = torch.cat(output_data[key_boneless])

    # Multiply image * lung
    torchresize = tvtransforms.Resize(image_spatial_size, interpolation=TF.InterpolationMode.NEAREST)
    if flag_segmentLung:
        masked = output_data["image"]*output_data["mask"]
        # Crop to lung and resize
        if flag_cropping:
            output_data["croppedMaskedImage"] =[]
            output_data["croppedMask"] =[]
            output_data["croppedImage"] =[]
            output_data["cropped"+key_boneless]=[]
            for idx2, mask_image in enumerate(output_data["mask"]):
                bb = BoundingBox(mask_image)
                indices = bb.findBox()
                # crop the mask & the masked image
                croppedMaskedImage = masked[idx2,:,indices["topbottom"][0]:indices["topbottom"][1]+1, indices["leftright"][0]:indices["leftright"][1]+1]
                croppedMask = output_data["mask"][idx2,:,indices["topbottom"][0]:indices["topbottom"][1]+1, indices["leftright"][0]:indices["leftright"][1]+1]
                croppedImage = output_data["image"][idx2,:,indices["topbottom"][0]:indices["topbottom"][1]+1, indices["leftright"][0]:indices["leftright"][1]+1]
                if key_boneless in data.keys():
                    croppedBoneless = output_data[key_boneless][idx2,:,indices["topbottom"][0]:indices["topbottom"][1]+1, indices["leftright"][0]:indices["leftright"][1]+1]
                # Resize image to 256x256
                output_data["croppedMaskedImage"].append(torchresize(croppedMaskedImage))
                output_data["croppedMask"].append(torchresize(croppedMask))
                output_data["croppedImage"].append(torchresize(croppedImage))
                if key_boneless in data.keys():
                    output_data["cropped"+key_boneless].append(torchresize(croppedBoneless))
            output_data["croppedMaskedImage"] = torch.stack(output_data["croppedMaskedImage"])
            output_data["croppedMask"] = torch.stack(output_data["croppedMask"])
            output_data["croppedImage"] = torch.stack(output_data["croppedImage"])
            if key_boneless in data.keys():
                output_data["cropped"+key_boneless] = torch.stack(output_data["cropped"+key_boneless])

            # IMPORTANT DATA OUTPUT
            maskedImage = output_data["croppedMaskedImage"]
        else:
            maskedImage = torchresize(masked)
    else:
        # No lung segmentation
        maskedImage = torchresize(output_data["image"])
    
    # Normalise masked images
    if flag_normalise:
        normalised = []
        for image in maskedImage:
            image = image.squeeze().numpy()
            image = normalisation(image)
            image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0) #[1x1xHxW]
            normalised.append(image)
        normalised = torch.cat(normalised)
        input_data = normalised
    else:
        input_data = maskedImage
    return input_data, output_data

def compositeImage(out, output_data, flag_cropping):
    # Re-paste the suppressed lung segment back into the OG image
    image_spatial_size = (out.shape[-2],out.shape[-1])
    torchresize = tvtransforms.Resize(image_spatial_size, interpolation=TF.InterpolationMode.NEAREST)
    
    if flag_cropping:
        mask = output_data["croppedMask"]
        OG_image = output_data["croppedImage"]
    else:
        mask = output_data["mask"]
        OG_image = output_data["image"]
        mask = torchresize(mask)
        OG_image = torchresize(OG_image)
    
    composited = []
    for minibatch_idx, lung in enumerate(out):
        mask_current = mask[minibatch_idx,:] #[CxHxW]
        # flip mask_current's Trues to Falses and vice versa
        body_mask = ~mask_current
        body = body_mask*OG_image[minibatch_idx,:]
        if body.shape[-2] != lung.shape[-2] and body.shape[-1] != lung.shape[-1]:
            body = torchresize(body)
        composited.append(body + lung)
    out = torch.stack(composited)
    if image_spatial_size is not None:
        out = torchresize(out)
    return out


# Utility Classes
class BoundingBox():
    """assumes an input torch tensor [1xHxW]"""
    def __init__(self, mask):
        self.mask = mask
    def findIndices(self, data_vector):
        # image is assumed to be [Nx1]
        indices = np.array((2,1))
        # find the index for the top and bottom of the data vector    
        first_index = np.nonzero(data_vector)
        first_index = first_index[0][0]
        last_index = np.nonzero(data_vector)[0][-1]
        indices[0] = first_index
        indices[1] = last_index
        return indices
    def findBox(self):
        # find mask's row & column extents
        mask = self.mask.detach().clone().squeeze().numpy()
        height = mask.shape[-2]
        width = mask.shape[-1]
        collapsed_mask_heightwise = np.sum(mask,-2)
        collapsed_mask_widthwise = np.sum(mask,-1)
        
        # left/right & top/bottom
        leftright = self.findIndices(collapsed_mask_heightwise)
        topbottom = self.findIndices(collapsed_mask_widthwise)
        return {"leftright":leftright, "topbottom":topbottom}