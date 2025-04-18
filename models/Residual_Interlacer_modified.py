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
from interlacer_layer_modified import *
from torchmetrics.functional import structural_similarity_index_measure
    
class ResidualInterlacerModified(nn.Module):
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
        
        self.interlacer_layers = nn.ModuleList([Interlacer_Modified(features_img=self.num_features_img,
                                  features_kspace=self.num_features_space,
                                  kernel_size=self.kernel_size,
                                  num_convs=self.num_convs,
                                  use_norm=self.use_norm,
                                 ) for i in range(self.num_layers)])
        
        
        #### Final convolutions for output
        
        self.conv1d_img = nn.Conv3d(in_channels=2, # this is 2*2, 2 for channels in image space, the other 2 because of torch.cat((img_in, inputs_img), dim=1) below
                                   out_channels=2, # 2 output channels in image space
                                   kernel_size=self.kernel_size, #self.kernel_size,
                                   padding='same')
        self.conv1d_kspace = nn.Conv3d(in_channels=2,
                                       out_channels=2,
                                       kernel_size=self.kernel_size, #self.kernel_size,
                                       padding='same')
        

    def forward(self, x):
        img_in, freq_in = x
        
        for i in range(self.num_layers):
            img_conv, k_conv = self.interlacer_layers[i]((img_in, freq_in))
            
            img_in = img_conv + img_in #inputs_img
            freq_in = k_conv + freq_in #inputs_kspace
        
        
        outputs_img = self.conv1d_img(img_in)
        outputs_kspace = self.conv1d_kspace(freq_in)
        
        #outputs_img = img_in    #self.conv1d_img(torch.cat((img_in, inputs_img), dim=1))
        #outputs_kspace = freq_in#self.conv1d_kspace(torch.cat((freq_in, inputs_kspace), dim=1))
        
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
    """
    Train the model for one epoch, applying a mask to both predictions and ground truth 
    before computing the loss.
    """
    model.train()  # Set model to training mode
    
    total_loss = 0.0
    num_samples = 0
    
    for inputs, dat_out, mask, _ in data_loader:
        # Unpack the inputs tuple
        img_in, freq_in = inputs
        # Move inputs and targets to the device
        img_in = img_in.to(device)
        freq_in = freq_in.to(device)
        dat_out = dat_out.to(device)
        mask = mask.to(device)
        
        # Prepare the input tuple for the model
        dat_in = (img_in, freq_in)
        
        optimizer.zero_grad()
        
        # Forward pass: model should accept tuple (img_in, freq_in)
        predictions, _ = model(dat_in)
        
        # Apply the mask to both predictions and ground truth
        masked_predictions = predictions * mask
        masked_targets = dat_out * mask
        
        # Compute loss on the masked data
        loss_curr = loss_fn(masked_predictions, masked_targets)
        
        # Backpropagation
        loss_curr.backward()
        optimizer.step()
        
        # Accumulate loss
        batch_size = dat_out.size(0)
        total_loss += loss_curr.item() * batch_size
        num_samples += batch_size
    
    # Return the average loss for the
    # Return the average loss for the dataset
    return total_loss / num_samples

def validate_model(model, loss_fn, data_loader, device, psnr_metric=None, ssim_metric=None):
    """Validation loop, computing average loss, PSNR, SSIM."""
    model.eval()
    total_loss = 0.0
    num_samples = 0

    # We'll compute the sum of PSNR and SSIM and then average
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0

    with torch.no_grad():
        for dat_in, dat_out, mask, norm_value in data_loader:
            inputs_img, inputs_kspace = [x.to(device) for x in dat_in]
            dat_out = dat_out.to(device)
            dat_in = (inputs_img, inputs_kspace)
            mask     = mask.to(device)
            norm_value = norm_value.to(device)
            norm_value = norm_value.view(norm_value.size(0), 1, 1, 1, 1)

            predictions, _ = model(dat_in)
            # Apply the mask to both predictions and ground truth
            masked_predictions = predictions * mask * norm_value
            masked_targets     = dat_out * mask * norm_value
            
            loss_curr = loss_fn(masked_predictions, masked_targets)

            batch_size = dat_out.size(0)
            total_loss += loss_curr.item() * batch_size
            num_samples += batch_size

            # If metrics are provided, calculate them
            if psnr_metric is not None:
                # The PSNR and SSIM metrics can handle batch tensors directly
                psnr_val = psnr_metric(predictions, dat_out)
                total_psnr += psnr_val.item()

            if ssim_metric is not None:
                ssim_val = ssim_metric(predictions, dat_out)
                total_ssim += ssim_val.item()

            num_batches += 1

    avg_loss = total_loss / num_samples

    # Compute average metrics if they exist
    avg_psnr = total_psnr / num_batches if num_batches > 0 else 0
    avg_ssim = total_ssim / num_batches if num_batches > 0 else 0

    return avg_loss, avg_psnr, avg_ssim



    
