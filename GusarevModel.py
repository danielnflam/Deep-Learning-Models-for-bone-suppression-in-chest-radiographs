import torch.nn as nn
import torch
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import os, sys, time, datetime, pathlib, random, math

##############################
# Gusarev Model
##############################

class Autoencoder(nn.Module):
    def __init__(self, input_array_size):
        super().__init__()
        self.input_array_size = input_array_size
        self.kernel_size = 5
        self.stride = 2
        self.padding = 2
        self.output_padding_convT = 1
        self.use_bias = True
        
        in_nc = input_array_size[1]
        out_nc = 16
        self.enc1 = nn.Conv2d(in_nc, out_nc, self.kernel_size, self.stride, self.padding, bias=self.use_bias)
        self.enc2 = nn.Conv2d(out_nc, out_nc*2, self.kernel_size, self.stride, self.padding, bias=self.use_bias)
        self.enc3 = nn.Conv2d(out_nc*2, out_nc*4, self.kernel_size, self.stride, self.padding, bias=self.use_bias)
        self.dec3 = nn.ConvTranspose2d(out_nc*4, out_nc*2, self.kernel_size, self.stride, self.padding, output_padding=self.output_padding_convT, bias=self.use_bias )
        self.dec2 = nn.ConvTranspose2d(out_nc*2, out_nc, self.kernel_size, self.stride, self.padding, output_padding=self.output_padding_convT, bias=self.use_bias )
        self.dec1 = nn.ConvTranspose2d(out_nc, in_nc , self.kernel_size, self.stride, self.padding, output_padding=self.output_padding_convT, bias=self.use_bias )
        
        # relu
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.enc1(x)
        out = self.relu(out)
        out = self.enc2(out)
        out = self.relu(out)
        out = self.enc3(out)
        out = self.relu(out)
        out = self.dec3(out)
        out = self.relu(out)
        out = self.dec2(out)
        out = self.relu(out)
        out = self.dec1(out)
        out = self.relu(out)
        return out