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
import torch.nn.functional as F
import random
import math
#from Unet import *

######## START AUGMENTATION ##########
import torch
import torch.nn.functional as F
import random
import math


    
    
###### STOP AUGMENTATION


class TensorDataset(Dataset):
    def __init__(self, data, labels, masks, gm_masks, wm_masks):
        """
        Initializes the dataset.

        Parameters:
            data: numpy array of shape (N, 2*grouped_time_steps, 22, 22, 21) for input data.
            labels: numpy array of shape (N, 2*grouped_time_steps, 22, 22, 21) for ground truth labels.
            masks: numpy array of shape (N, 2, 22, 22, 21) with 0s and 1s, used to mask out noise.
            gm_masks: numpy array of shape (N, 1, 22, 22, 21) indicating gray matter regions.
            wm_masks: numpy array of shape (N, 1, 22, 22, 21) indicating white matter regions.
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.float32)
        self.gm_masks = torch.tensor(gm_masks, dtype=torch.float32)
        self.wm_masks = torch.tensor(wm_masks, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]       # Shape: (2*grouped_time_steps, 22, 22, 21)
        label = self.labels[idx]     # Shape: (2*grouped_time_steps, 22, 22, 21)
        mask = self.masks[idx]       # Shape: (2, 22, 22, 21)
        gm_mask = self.gm_masks[idx] # Shape: (1, 22, 22, 21)
        wm_mask = self.wm_masks[idx] # Shape: (1, 22, 22, 21)
        return data, label, mask, gm_mask, wm_mask
    

import torch
import torch.nn as nn
import torch.nn.functional as F

############################
# 3D Double Conv (LeakyReLU)
############################


def double_conv_3d(in_channels, out_channels, use_batch_norm=False):
    """
    Two consecutive 3D convolutions + optional batch norm + LeakyReLU.
    """
    layers = []
    layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
    if use_batch_norm:
        layers.append(nn.BatchNorm3d(out_channels))
    layers.append(nn.LeakyReLU(inplace=True))
    
    layers.append(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1))
    if use_batch_norm:
        layers.append(nn.BatchNorm3d(out_channels))
    layers.append(nn.LeakyReLU(inplace=True))
    
    return nn.Sequential(*layers)



############################
# Single UNet3D (4→2) w/ skip
############################
class UNet3D(nn.Module):
    """
    A flexible 3D U-Net that can be used for:
      - 4 -> 4 with a "full" global skip (out += x), or
      - 4 -> 2 with a "first2" skip (out += x[:, :2]), or
      - 4 -> 2 / 4 -> 4 with no skip if desired.

    The encoder/decoder channel sizes (64,128,256,512) are the same
    as in the original example, but you can adjust them as needed.
    """
    def __init__(self, in_channels, out_channels,
                 use_batch_norm=False, skip_mode="none"):
        """
        skip_mode: one of {"none", "full", "first2"}.
          - "full":  in_channels == out_channels, and we do (out += x).
          - "first2": out_channels == 2, and we do (out += x[:, :2]).
          - "none": no skip connection.
        """
        super(UNet3D, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_mode = skip_mode
        
        # ----------- Encoder (Downsampling) -----------
        self.enc1 = double_conv_3d(in_channels, 64, use_batch_norm)
        self.enc2 = double_conv_3d(64, 128, use_batch_norm)
        self.enc3 = double_conv_3d(128, 256, use_batch_norm)
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # ----------- Bottleneck -----------
        self.bottleneck = double_conv_3d(256, 512, use_batch_norm)
        
        # ----------- Decoder (Upsampling) -----------
        self.up3 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec3 = double_conv_3d(256 + 256, 256, use_batch_norm)
        
        self.up2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec2 = double_conv_3d(128 + 128, 128, use_batch_norm)
        
        self.up1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec1 = double_conv_3d(64 + 64, 64, use_batch_norm)
        
        # Final 3D conv
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # ---------------- Encoder ----------------
        e1 = self.enc1(x)      # (B,64, ...)
        p1 = self.pool(e1)     # (B,64, ...)
        
        e2 = self.enc2(p1)     # (B,128, ...)
        p2 = self.pool(e2)     # (B,128, ...)
        
        e3 = self.enc3(p2)     # (B,256, ...)
        p3 = self.pool(e3)     # (B,256, ...)
        
        # ---------------- Bottleneck ----------------
        b = self.bottleneck(p3)# (B,512, ...)
        
        # ---------------- Decoder ----------------
        u3 = self.up3(b)              # (B,256, ...)
        cat3 = torch.cat([u3, e3], 1) # (B,512, ...)
        d3 = self.dec3(cat3)          # (B,256, ...)
        
        u2 = self.up2(d3)             # (B,128, ...)
        cat2 = torch.cat([u2, e2], 1) # (B,256, ...)
        d2 = self.dec2(cat2)          # (B,128, ...)
        
        u1 = self.up1(d2)             # (B,64, ...)
        cat1 = torch.cat([u1, e1], 1) # (B,128, ...)
        d1 = self.dec1(cat1)          # (B,64, ...)
        
        out = self.final_conv(d1)     # (B, out_channels, ...)
        
        # ------------ Global Skip Logic ------------
        if self.skip_mode == "full":
            # add the entire x to out => requires x.shape == out.shape
            out = out + x
        elif self.skip_mode == "first2":
            # add only first 2 channels of x => requires out_channels==2
            # so x[:, :2] should match out in shape
            out = out + x[:, :2]
        # else: skip_mode="none", do nothing
        
        return out



##############################
# Stacked UNet3D (4→2) * N
##############################
class StackedUNet3D(nn.Module):
    """
    Stacks N copies of UNet3D:
      - For the first N-1, each UNet is (4->4) with full skip.
      - For the final UNet, (4->2) with a partial skip that adds x[:, :2].
    """
    def __init__(self, num_unets=2, use_batch_norm=False):
        super(StackedUNet3D, self).__init__()
        
        self.num_unets = num_unets
        self.unets = nn.ModuleList()
        
        # First N-1 unets: 4->4 with full skip
        for i in range(num_unets - 1):
            self.unets.append(
                UNet3D(in_channels=4, out_channels=4,
                       use_batch_norm=use_batch_norm,
                       skip_mode="full")   # 4->4 w/ full skip
            )
        
        # Final unet: 4->2 with skip_mode="first2"
        self.unets.append(
            UNet3D(in_channels=4, out_channels=2,
                   use_batch_norm=use_batch_norm,
                   skip_mode="first2")   # 4->2 w/ partial skip => out += x[:, :2]
        )
        
    def forward(self, x):
        """
        x should have shape (B, 4, D, H, W) at the start.
        (Typically 2 data channels + 2 coords, or any 4 channels.)
        """
        for i, unet in enumerate(self.unets):
            x = unet(x)  # shape will be 4->4 except on last pass (4->2)
            
            # After final UNet, x will have 2 channels => done
            if i < self.num_unets - 1:
                # x remains 4 channels (since out_channels=4 on these blocks)
                pass
            else:
                # x is now 2 channels after the final UNet
                return x

        # Just in case we never "return" (we do above), we return here too
        return x






######################
# Custom Loss (no C2)#
######################
class CustomLoss(nn.Module):
    def __init__(self, l1_lambda=0.0):
        """
        Args:
            l1_lambda (float): Regularization strength for model weights (if desired).
        """
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_lambda = l1_lambda

    def forward(self, outputs, targets, model=None):
        """
        Compute MSE + optional L1 weight regularization (no C2).
        """
        mse_loss = self.mse_loss(outputs, targets)
        if self.l1_lambda > 0.0 and model is not None:
            l1_loss_model = 0
            for param in model.parameters():
                l1_loss_model += torch.sum(torch.abs(param))
            return mse_loss + self.l1_lambda * l1_loss_model
        else:
            return mse_loss


###################################
# Training loop (no C2 references)#
###################################
def train_one_epoch(model, optimizer, loss_fn, data_loader, device='cpu'):
    """
    Trains the YNet for one epoch.
    
    Args:
        model (nn.Module): The updated UNet3D (YNet) that accepts 4-channel input.
        optimizer (torch.optim.Optimizer): Optimizer.
        loss_fn (callable): Loss function (expects masked_predictions, masked_targets, model, C2).
        data_loader (DataLoader): Yields tuples: (dat_in, dat_out, mask, gm_mask, wm_mask).
        device (str): 'cpu' or 'cuda'.
    
    Returns:
        float: Average loss for the epoch.
    """
    model.train()  # Set model to training mode
    total_loss = 0.0
    num_samples = 0

    for dat_in, dat_out, mask, gm_mask, wm_mask in data_loader:
        # Move inputs, outputs, and masks to device
        dat_in  = dat_in.to(device)    # (N, 2*k, H, W, D)
        dat_out = dat_out.to(device)   # (N, 2*k, H, W, D)
        mask    = mask.to(device)      # (N, 2, H, W, D)
        gm_mask = gm_mask.to(device)    # (N, 1, H, W, D)
        wm_mask = wm_mask.to(device)    # (N, 1, H, W, D)
        
        # Concatenate the original spectral data with the extra masks along the channel axis.
        # New input shape: (N, 2*k + 2, H, W, D)
        inputs = torch.cat([dat_in, gm_mask, wm_mask], dim=1)
        
        optimizer.zero_grad()  # Clear previous gradients
        
        predictions = model(inputs)  # Forward pass
        
        # --- Optionally crop to original size if needed (e.g., to remove padding) ---
        # For example:
        # predictions_cropped = predictions[:, :, 1:-1, 1:-1, 1:-2]
        # targets_cropped = dat_out[:, :, 1:-1, 1:-1, 1:-2]
        # mask_cropped = mask[:, :, 1:-1, 1:-1, 1:-2]
        # Here we'll use the full tensors:
        predictions_cropped = predictions
        targets_cropped = dat_out
        mask_cropped = mask
        
        # Compute loss on the masked region
        masked_predictions = predictions_cropped * mask_cropped
        masked_targets = targets_cropped * mask_cropped
        loss_curr = loss_fn(masked_predictions, masked_targets, model)
        
        loss_curr.backward()  # Backpropagation
        optimizer.step()  # Update model parameters
        
        batch_size = dat_in.size(0)
        total_loss += loss_curr.item() * batch_size
        num_samples += batch_size
        
    return total_loss / num_samples

def validate_model(model, loss_fn, data_loader, device='cpu'):
    """
    Validation loop to compute the average loss.
    
    Args:
        model (nn.Module): The updated YNet model.
        loss_fn (callable): Loss function (expects masked_predictions, masked_targets, model, C2).
        data_loader (DataLoader): Yields tuples: (dat_in, dat_out, mask, gm_mask, wm_mask).
        device (str): 'cpu' or 'cuda'.
    
    Returns:
        float: Average loss over the validation set.
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for dat_in, dat_out, mask, gm_mask, wm_mask in data_loader:
            # Move tensors to device
            dat_in  = dat_in.to(device)    # (N, 2*k, H, W, D)
            dat_out = dat_out.to(device)
            mask    = mask.to(device)      # (N, 2, H, W, D)
            gm_mask = gm_mask.to(device)    # (N, 1, H, W, D)
            wm_mask = wm_mask.to(device)    # (N, 1, H, W, D)
            
            # Concatenate inputs with masks
            inputs = torch.cat([dat_in, gm_mask, wm_mask], dim=1)
            
            # Forward pass
            predictions = model(inputs)  # (N, 2*k, H, W, D)
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            
            # Optionally crop outputs if necessary; here we use full tensors:
            predictions_cropped = predictions
            targets_cropped = dat_out
            mask_cropped = mask
            
            # Apply the mask to focus on valid regions
            masked_predictions = predictions_cropped * mask_cropped
            masked_targets = targets_cropped * mask_cropped
            
            loss_curr = loss_fn(masked_predictions, masked_targets, model)
            
            batch_size = dat_in.size(0)
            total_loss += loss_curr.item() * batch_size
            num_samples += batch_size

    return total_loss / num_samples if num_samples > 0 else 0.0





def compute_mse(model, data_loader, device='cpu'): # I like to always keep track of MSE, to see if my model is objectivly improving when regularizing the loss function 
    model.eval()  # Set model to evaluation mode
    mse_loss_fn = torch.nn.MSELoss()
    
    total_mse = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for dat_in, dat_out in data_loader:
            dat_in, dat_out = dat_in.to(device), dat_out.to(device)  # Move to device
            predictions, _ = model(dat_in)  # Forward pass
            mse_loss = mse_loss_fn(predictions, dat_out)  # Compute MSE loss
            total_mse += mse_loss.item() * dat_in.size(0)  # Accumulate MSE for this batch
            num_samples += dat_in.size(0)  # Count samples
    
    # Return the average MSE for the dataset
    return total_mse / num_samples


def resize_images_zoom(images, target_size): # this function converts the image to a smaller size, to speed up training etc
    # Calculate the zoom factor
    zoom_factor = target_size / images.shape[-1]
    
    # Create an empty array for the resized images
    num_samples, num_channels, height, width = images.shape
    resized_images = np.empty((num_samples, num_channels, target_size, target_size))
    
    # Resize each image
    for i in range(num_samples):
        for j in range(num_channels):
            resized_images[i, j] = zoom(images[i, j], zoom_factor, order=3)  # Cubic interpolation
    
    return resized_images


def compute_fourier_transform_5d(images):
    """
    Computes the 2D Fourier Transform of each 'image' in a 5D array along the first two dimensions.
    
    Parameters
    ----------
    images : np.ndarray
        A 5D numpy array of shape (N1, N2, N3, N4, N5), where (N1, N2) corresponds to 
        the spatial dimensions of each 2D image, and (N3, N4, N5) are additional dimensions.
        The entries may be real or complex.
        
    Returns
    -------
    f_transform_shifted : np.ndarray
        A numpy array of the same shape (N1, N2, N3, N4, N5), containing complex values of the 
        Fourier-transformed images with zero frequencies centered.
    """
    # Perform a 2D FFT along the first two dimensions
    f_transform = np.fft.fft2(images, axes=(0, 1, 2))
    
    # Shift the zero frequency component to the center
    f_transform_shifted = np.fft.fftshift(f_transform, axes=(0, 1, 2))
    
    return f_transform_shifted

def compute_inverse_fourier_transform_5d(fourier_data):
    """
    Computes the 2D Inverse Fourier Transform of each 'image' in a 5D array along the first two dimensions.
    
    Parameters
    ----------
    fourier_data : np.ndarray
        A 5D numpy array of complex Fourier data with shape (N1, N2, N3, N4, N5).
        
    Returns
    -------
    inverse_transform : np.ndarray
        A numpy array of the same shape (N1, N2, N3, N4, N5), containing the inverse Fourier
        transformed images as complex values.
    """
    # Shift the zero frequency component back to the original positions
    shifted_data = np.fft.ifftshift(fourier_data, axes=(0, 1, 2))
    
    # Perform the 2D Inverse FFT along the first two dimensions
    inverse_transform = np.fft.ifft2(shifted_data, axes=(0, 1, 2))
    
    return inverse_transform

import numpy as np

import numpy as np

def undersample_FT_data(fourier_data, k, U):
    """
    Undersamples the 3D Fourier-transformed data in k-space with less undersampling near the center,
    while incorporating randomness.

    A fraction U (between 0 and 1) of entries in the 3D k-space (first three dimensions)
    will be set to zero, prioritizing retention of entries near the center of k-space.

    Parameters
    ----------
    fourier_data : np.ndarray
        A numpy array of complex Fourier data with shape:
        (X, Y, Z, ...) 
        where X, Y, Z are spatial dimensions, and ... indicates zero or more additional dimensions.
    
    U : float
        The fraction of entries to set to zero (undersample). Must be between 0 and 1.
        For example:
        - U = 0.5 will zero out 50% of the entries.
        - U = 0.1 will zero out 10% of the entries.

    Returns
    -------
    undersampled_data : np.ndarray
        A numpy array of the same shape as `fourier_data`, with approximately a fraction U of entries
        in the first three dimensions set to zero.
    """
    if not (0 <= U <= 1):
        raise ValueError("U must be between 0 and 1.")
    
    if k == 1:
        # Shape of the k-space
        shape = fourier_data.shape[:3]
        center = np.array(shape) // 2  # Center of k-space (approximate)

        # Create a grid of distances from the center
        X, Y, Z = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij'
        )
        distances = np.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)

        # Normalize distances to [0, 1]
        max_distance = np.max(distances)
        normalized_distances = distances / max_distance

        # Compute probabilities with less undersampling near the center
        decay_factor = 7  # Controls how steeply probabilities decay from the center
        probabilities = np.exp(-decay_factor * normalized_distances**2)

        # Generate a random mask based on probabilities
        mask = np.random.rand(*shape) < probabilities

        # Adjust the mask to ensure the global undersampling rate matches U
        total_entries = np.prod(shape)
        num_keep = int((1 - U) * total_entries)  # Number of entries to retain
        current_keep = np.sum(mask)
        if current_keep > num_keep:
            # Too many points retained, randomly set excess to zero
            excess = current_keep - num_keep
            indices_to_zero = np.random.choice(np.flatnonzero(mask), size=excess, replace=False)
            mask.flat[indices_to_zero] = False
        elif current_keep < num_keep:
            # Too few points retained, randomly set additional points to one
            deficit = num_keep - current_keep
            indices_to_keep = np.random.choice(np.flatnonzero(~mask), size=deficit, replace=False)
            mask.flat[indices_to_keep] = True

        # Extend mask to match the shape of the full array
        mask = mask[..., np.newaxis]

        # Apply the mask to the Fourier data
        undersampled_data = fourier_data * mask

    else:
        # Shape of the k-space
        shape = fourier_data.shape[:3]
        num_channels = fourier_data.shape[3:]  # Remaining dimensions after k-space

        # Divide indices into k complementary masks
        masks = np.zeros((shape[0], shape[1], shape[2], k), dtype=bool)
        # Explicitly loop over the indices of the 3D array
        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    random_k = np.random.randint(0, k)  # Random integer between 0 and k-1
                    masks[x, y, z, random_k] = 1

        # Apply the masks to the Fourier data
        # Broadcasting masks along the additional dimensions of fourier_data
        undersampled_data = np.zeros_like(fourier_data)
        
        Last_dim = undersampled_data.shape[-1]
        
        for m in range(0,Last_dim):
            undersampled_data[..., m] = fourier_data[..., m] * masks
        
    return undersampled_data






















    
