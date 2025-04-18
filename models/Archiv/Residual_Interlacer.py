import os
import torch
from torch import nn
import numpy as np
import time
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage import zoom # for compressing images / only for testing purposes to speed up NN training
from scipy.fft import fft2, fftshift
from torch.utils.data import Dataset, DataLoader
from interlacer_layer import *
from torchmetrics.functional import structural_similarity_index_measure
    
class ResidualInterlacer(nn.Module):
    def __init__(self, kernel_size,
                        num_features_img,
                        num_features_kspace,
                        num_convs,
                        num_layers,
                        use_norm):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_features_img = num_features_img
        self.num_features_space = num_features_kspace
        self.num_convs = num_convs
        self.num_layers = num_layers
        self.use_norm = use_norm
        
        self.interlacer_layers = nn.ModuleList([Interlacer(features_img=self.num_features_img,
                                  features_kspace=self.num_features_space,
                                  kernel_size=self.kernel_size,
                                  num_convs=self.num_convs,
                                  use_norm=self.use_norm,
                                 ) for i in range(self.num_layers)])
        
        
        self.conv1d_img = nn.Conv3d(in_channels=4, # this is 2*2, 2 for channels in image space, the other 2 because of torch.cat((img_in, inputs_img), dim=1) below
                                   out_channels=2, # 2 output channels in image space
                                   kernel_size=self.kernel_size, #self.kernel_size,
                                   padding='same')
        self.conv1d_kspace = nn.Conv3d(in_channels=4,
                                       out_channels=2,
                                       kernel_size=self.kernel_size, #self.kernel_size,
                                       padding='same')
        

    def forward(self, x):
        inputs_img, inputs_kspace = x
        
        img_in = inputs_img
        freq_in = inputs_kspace
        
        for i in range(self.num_layers):
            img_conv, k_conv = self.interlacer_layers[i]((img_in, freq_in, inputs_img, inputs_kspace))
            
            img_in = img_conv + img_in #inputs_img
            freq_in = k_conv + freq_in #inputs_kspace
        
        outputs_img = self.conv1d_img(torch.cat((img_in, inputs_img), dim=1))
        outputs_kspace = self.conv1d_kspace(torch.cat((freq_in, inputs_kspace), dim=1))
        

        return (outputs_img, outputs_kspace)


class CustomLoss(nn.Module):
    """Loss function combining MSE and SSIM, as in Pauls paper."""
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs, targets):
        """
        Args:
          outputs: Predicted images from the model.
          targets: Ground truth images.
        Returns:
          Combined MSE and SSIM loss.
        """
        # Compute MSE loss
        mse = self.mse_loss(outputs, targets)
        
        # Compute SSIM
        #ssim = structural_similarity_index_measure(outputs, targets, data_range=1.0)  # Ensure normalized inputs
        
        # Combine losses: L = MSE + (1 - SSIM)
        combined_loss = mse #+ 0.1*(1 - ssim)
        return combined_loss
    
def train_one_epoch(model, optimizer, loss_fn, data_loader, device='cpu'):
    model.train()  # Set model to training mode
    
    total_loss = 0.0
    num_samples = 0
    
    for dat_in, dat_out in data_loader:
        # Convert the list into a tuple of tensors and move to device
        inputs_img, inputs_kspace = [x.to(device) for x in dat_in]
        dat_out = dat_out.to(device)

        # Prepare the input tuple
        dat_in = (inputs_img, inputs_kspace)
        
        optimizer.zero_grad()  # Clear previous gradients
        predictions, _ = model(dat_in)  # Forward pass (ignore k-space output)
        
        loss_curr = loss_fn(predictions, dat_out)  # Compute loss
        loss_curr.backward()  # Compute gradients
        optimizer.step()  # Update model weights
        
        total_loss += loss_curr.item() * dat_out.size(0)  # Accumulate loss for this batch
        num_samples += dat_out.size(0)  # Count samples in this batch
    
    # Return the average loss for the dataset
    return total_loss / num_samples

def validate_model(model, loss_fn, data_loader, device='cpu'):
    """
    Validates the model on the given DataLoader, computing MSE loss only in image space.
    Args:
        model: The model to validate.
        loss_fn: The loss function to compute the error.
        data_loader: DataLoader for the validation dataset.
        device: Device to run the validation on ('cpu' or 'cuda').

    Returns:
        Average validation loss for the dataset.
    """
    model.eval()  # Set model to evaluation mode
    
    total_loss = 0.0
    num_samples = 0
    
    with torch.no_grad():  # Disable gradient computation
        for dat_in, dat_out in data_loader:
            # Convert the list into a tuple of tensors and move to device
            inputs_img, inputs_kspace = [x.to(device) for x in dat_in]
            dat_out = dat_out.to(device)

            # Prepare the input tuple
            dat_in = (inputs_img, inputs_kspace)
            
            # Forward pass
            predictions, _ = model(dat_in)  # Ignore k-space output
            
            # Compute loss only in image space
            loss_curr = loss_fn(predictions, dat_out)  # Compute MSE between predictions and ground truth
            
            # Accumulate loss and sample count
            total_loss += loss_curr.item() * dat_out.size(0)
            num_samples += dat_out.size(0)
    
    # Return the average loss
    return total_loss / num_samples



    
