import torch.nn as nn
import torch
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import os, sys, time, datetime, pathlib, random, math

##############################
# Gusarev Model
##############################
# AutoEncoder
class Autoencoder(nn.Module):
    def __init__(self, input_array_size):
        super().__init__()
        print("Using AutoEncoder Model.")
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
        
        # output layer
        self.output_layer = nn.Sigmoid()
        
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
        return self.output_layer(out)
# 6 Layer CNN Model
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bias):
        super().__init__()
        self.use_bias = use_bias
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=self.use_bias)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        
        return out

class MultilayerCNN(nn.Module):
    def __init__(self, input_array_size):
        super().__init__()
        print("Using 6-Layer MultiCNN Model.")
        self.input_array_size = input_array_size
        self.kernel_size = 5
        self.stride = 1
        self.padding = 2
        self.use_bias = True
        
        in_nc = input_array_size[1]
        out_nc = 16
        self.layer1 = ConvBlock(in_nc, out_nc, self.kernel_size, self.stride, self.padding, use_bias=self.use_bias)
        self.layer2 = ConvBlock(out_nc, out_nc*2, self.kernel_size, self.stride, self.padding, use_bias=self.use_bias)
        self.layer3 = ConvBlock(out_nc*2, out_nc*4, self.kernel_size, self.stride, self.padding, use_bias=self.use_bias)
        self.layer4 = ConvBlock(out_nc*4, out_nc*8, self.kernel_size, self.stride, self.padding, use_bias=self.use_bias)
        self.layer5 = ConvBlock(out_nc*8, out_nc*16, self.kernel_size, self.stride, self.padding, use_bias=self.use_bias)
        #self.layer6 = ConvBlock(out_nc*16, in_nc, self.kernel_size, self.stride, self.padding, use_bias=self.use_bias)

        # output layer
        self.conv_output = nn.Conv2d(out_nc*16, in_nc, self.kernel_size, self.stride, self.padding, use_bias=self.use_bias)
        self.output_layer = nn.Sigmoid()
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.conv_output(out)
        return self.output_layer(out)