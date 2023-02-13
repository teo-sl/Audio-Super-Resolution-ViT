import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel, AdamW

import os
import math
import glob
import pickle


import librosa
from scipy import signal

import matplotlib.pyplot as plt
import numpy as np


class ResidualBlock(nn.Module):

    def __init__(self, device='cpu'):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
                        nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(3,3), padding=1), 
                        nn.GELU(), 
                        nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(3,3), padding=1)
        ).to(device) 

        
    def forward(self, inputs):
        convolved_input = self.block(inputs)
        return convolved_input + inputs



class GenerativeNetwork(nn.Module):
    
    def __init__(self, device='cpu'):
        super(GenerativeNetwork, self).__init__()
        self.device = device
        self.hidden_size = 64
        self.patch_size = 16
        configuration = ViTConfig(num_attention_heads=8, num_hidden_layers=8, hidden_size=self.hidden_size, patch_size=self.patch_size, num_channels=1, image_size=1024)
        self.vit = ViTModel(configuration).to(self.device)
        self.model = nn.Sequential(
                        # bring the image back to the original size
                        nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, padding=1, stride=2), 
                        nn.GELU(), 
                      
                        # skip connections
                        ResidualBlock(),
                        nn.GELU(),                      
                        ResidualBlock(),
                        nn.GELU(),
                        ResidualBlock(),
                        nn.GELU(), 
                        ResidualBlock(),
                        nn.GELU(),  
        ).to(device)
        

    def patch_to_img(self, x, patch_size):
        B, NumPatches, HiddenSize = x.shape
        x = x.reshape(B, NumPatches, 1, HiddenSize)
        x = x.reshape(B, NumPatches, 1, patch_size, patch_size)
        x = x.permute(0, 1, 3, 4, 2)
        x = x.reshape(B, int(math.sqrt(NumPatches)), int(math.sqrt(NumPatches)), patch_size, patch_size, 1)
        x = x.permute(0,1,3,2,4,5)
        new_h = x.shape[1] * x.shape[2]
        new_w = x.shape[3] * x.shape[4]
        x = x.reshape(B, new_h, new_w, 1) #ultima posizione = num_channels che è sempre 1
        x = x.swapaxes(3, 1)
        x = x.swapaxes(3, 2)
        return x
    
        
    def forward(self, inputs):
        if inputs.device == 'cpu':
            inputs = inputs.to(self.device)
        vit_res = self.vit(pixel_values=inputs)
        inputs = vit_res.last_hidden_state[:, 1:, :]
        patch_size_after_vit = int(math.sqrt(inputs.shape[2]))
        inputs = self.patch_to_img(inputs, patch_size_after_vit)
        return self.model(inputs)