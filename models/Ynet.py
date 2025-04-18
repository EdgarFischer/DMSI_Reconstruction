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

######## START AUGMENTATION ##########
import torch
import torch.nn.functional as F
import random
import math


class TensorDatasetWithGM(Dataset):
    def __init__(self, data, labels, masks, gm_masks):
        """
        Initializes the dataset.

        Parameters:
            data: numpy array of shape (N, 2*grouped_time_steps, 22, 22, 21) for input data.
            labels: numpy array of shape (N, 2*grouped_time_steps, 22, 22, 21) for ground truth labels.
            masks: numpy array of shape (N, 2, 22, 22, 21) with 0s and 1s, used to mask out noise.
            gm_masks: numpy array of shape (N, 1, 22, 22, 21) indicating gray matter regions (1) and non-gray matter (0).
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.float32)
        self.gm_masks = torch.tensor(gm_masks, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]     # Shape: (2*grouped_time_steps, 22, 22, 21)
        label = self.labels[idx]   # Shape: (2*grouped_time_steps, 22, 22, 21)
        mask = self.masks[idx]     # Shape: (2, 22, 22, 21)
        gm_mask = self.gm_masks[idx]   # Shape: (1, 22, 22, 21)
        return data, label, mask, gm_mask

    
import torch
import torch.nn as nn
import torch.nn.functional as F

def double_conv_3d(in_channels, out_channels, use_batch_norm=True):
    """
    A helper function to create two consecutive 3D Conv + (optional) BatchNorm + LeakyReLU layers.
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

class Encoder3D(nn.Module):
    """
    Simple 3D Encoder with 3 levels + bottleneck.
    """
    def __init__(self, in_channels=2, use_batch_norm=True):
        super().__init__()
        
        self.enc1 = double_conv_3d(in_channels, 64, use_batch_norm)
        self.enc2 = double_conv_3d(64, 128, use_batch_norm)
        self.enc3 = double_conv_3d(128, 256, use_batch_norm)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.bottleneck = double_conv_3d(256, 512, use_batch_norm)
    
    def forward(self, x):
        e1 = self.enc1(x)       # (N, 64, ...)
        p1 = self.pool(e1)
        
        e2 = self.enc2(p1)      # (N, 128, ...)
        p2 = self.pool(e2)
        
        e3 = self.enc3(p2)      # (N, 256, ...)
        p3 = self.pool(e3)
        
        b = self.bottleneck(p3) # (N, 512, ...)
        
        return e1, e2, e3, b

class YNet3D(nn.Module):
    """
    A two-branch 3D U-Net-like architecture ("Y-Net") that fuses two encoders into a shared decoder.
    It incorporates standard skip connections from each encoder level to the corresponding decoder level
    and includes a residual connection from the very beginning (i.e. the sum of the two masked inputs)
    to the final output.
    """
    def __init__(self, 
                 in_channels_branchA=2,
                 in_channels_branchB=2,
                 use_batch_norm=False):
        super(YNet3D, self).__init__()
        
        # Two encoders (for example, one for GM-masked and one for WM-masked data)
        self.encA = Encoder3D(in_channels_branchA, use_batch_norm)
        self.encB = Encoder3D(in_channels_branchB, use_batch_norm)
        
        # Fuse bottlenecks: each encoder outputs 512 channels.
        # Concatenating gives 512*2 = 1024 channels.
        self.bottleneck_fuse = double_conv_3d(512*2, 512, use_batch_norm)
        
        # Decoder: Level 3
        # Upsample from bottleneck: 512 -> 256 channels.
        self.up3 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        # Skip connections: from encoder level 3 of both branches (256 each)
        # Total: 256 (upsampled) + 256 + 256 = 768 channels.
        self.dec3 = double_conv_3d(768, 256, use_batch_norm)
        
        # Decoder: Level 2
        self.up2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        # Skip: from encoder level 2 from both branches (128 each): 128 (upsampled) + 128 + 128 = 384.
        self.dec2 = double_conv_3d(384, 128, use_batch_norm)
        
        # Decoder: Level 1
        self.up1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        # Skip: from encoder level 1 from both branches (64 each): 64 (upsampled) + 64 + 64 = 192.
        self.dec1 = double_conv_3d(192, 64, use_batch_norm)
        
        # Final output: a 1x1 convolution to produce the desired number of output channels.
        self.final_conv = nn.Conv3d(64, 2, kernel_size=1)
        
    def forward(self, xA, xB):
        """
        Args:
            xA: Input for branch A (e.g. gray matter–masked), shape: (N, in_channels_branchA, X, Y, Z)
            xB: Input for branch B (e.g. white matter–masked), shape: (N, in_channels_branchB, X, Y, Z)
        Returns:
            out: Final reconstructed volume with residual connection, shape: (N, 2, X, Y, Z)
        """
        # Encoder pass for branch A
        e1A, e2A, e3A, bA = self.encA(xA)
        # Encoder pass for branch B
        e1B, e2B, e3B, bB = self.encB(xB)
        
        # Fuse bottlenecks: concatenate along the channel dimension
        bottleneck_cat = torch.cat([bA, bB], dim=1)  # (N, 1024, X/8, Y/8, Z/8)
        fused_b = self.bottleneck_fuse(bottleneck_cat)  # (N, 512, X/8, Y/8, Z/8)
        
        # Decoder: Level 3
        u3 = self.up3(fused_b)  # (N, 256, X/4, Y/4, Z/4)
        cat3 = torch.cat([u3, e3A, e3B], dim=1)  # (N, 256 + 256 + 256 = 768, X/4, Y/4, Z/4)
        d3 = self.dec3(cat3)   # (N, 256, X/4, Y/4, Z/4)
        
        # Decoder: Level 2
        u2 = self.up2(d3)  # (N, 128, X/2, Y/2, Z/2)
        cat2 = torch.cat([u2, e2A, e2B], dim=1)  # (N, 128 + 128 + 128 = 384, X/2, Y/2, Z/2)
        d2 = self.dec2(cat2)  # (N, 128, X/2, Y/2, Z/2)
        
        # Decoder: Level 1
        u1 = self.up1(d2)  # (N, 64, X, Y, Z)
        cat1 = torch.cat([u1, e1A, e1B], dim=1)  # (N, 64 + 64 + 64 = 192, X, Y, Z)
        d1 = self.dec1(cat1)  # (N, 64, X, Y, Z)
        
        # Final output
        out = self.final_conv(d1)  # (N, 2, X, Y, Z)
        
        # Residual connection: assume xA + xB gives the original input (if the masks sum to 1)
        residual = xA + xB  # (N, 2, X, Y, Z)
        out = out + residual
        
        return out


class CustomLoss(nn.Module):
    def __init__(self, l1_lambda=0.0):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_lambda = l1_lambda  # If you still want to do any extra penalty

    def forward(self, outputs, targets):
        # A simple MSE or can incorporate additional logic
        return self.mse_loss(outputs, targets)


class CustomLoss2(nn.Module):     # this is a modifed loss function that includes L1 regularization. I noticed that the network has a much larger error on validation then on training set; likely overfitting
    def __init__(self, l1_lambda_c2=0.0001, mse_lambda=1.0,):
        super(CustomLoss2, self).__init__()
        self.mse_loss = nn.MSELoss()  # Mean Squared Error loss
        self.mse_lambda = mse_lambda  # Lambda for the MSE loss (if needed)
        self.l1_lambda_c2 = l1_lambda_c2  # Lambda for L1 regularization on C2 layer

    def forward(self, outputs, targets, model, C2):
        # Compute the MSE loss
        mse_loss = self.mse_loss(outputs, targets)
        
        # Compute L1 regularization loss for the C2 layer
        l1_loss_c2 = self.l1_lambda_c2 * torch.sum(torch.abs(C2))
        
        # Compute L1 regularization term applied to all weights, to avoid overfitting
        l1_loss_model = 0
        for param in model.parameters():
            l1_loss_model += torch.sum(torch.abs(param))
        
        # Total loss combining MSE loss and L1 regularization
        total_loss = mse_loss + l1_loss_c2 + self.mse_lambda * l1_loss_model
        
        return total_loss    
      

import torch
import torch.nn.functional as F

def train_one_epoch_ynet(model, optimizer, loss_fn, data_loader, device="cuda"):
    """
    Trains the YNet model for one epoch.
    
    For each batch, multiplies the input by the gray matter mask (gm_mask)
    to get branch A, and by (1 - gm_mask) for branch B. Then performs
    a forward pass, computes the loss (optionally applying a general mask),
    and updates model weights.
    
    Args:
        model (nn.Module): The YNet model with signature model(xGM, xWM).
        optimizer (torch.optim.Optimizer): Optimizer for training.
        loss_fn (callable): Loss function, e.g. MSELoss (expects (predictions, targets)).
        data_loader (DataLoader): DataLoader yielding (inputs, targets, masks, gm_mask).
            - inputs: shape (N, 2, z, f, T)
            - targets: shape (N, 2, z, f, T)
            - masks: shape (N, 2, z, f, T) (for overall masking if needed)
            - gm_mask: shape (N, 1, z, f, T) with 1 for gray matter, 0 elsewhere.
        device (str): 'cpu' or 'cuda'.
    
    Returns:
        float: Average loss over the epoch.
    """
    model.train()
    epoch_loss = 0.0
    num_samples = 0

    for batch_idx, (inputs, targets, masks, gm_mask) in enumerate(data_loader):
        # Move inputs to device
        inputs  = inputs.to(device)    # shape: (N, 2, z, f, T)
        targets = targets.to(device)
        masks   = masks.to(device)
        gm_mask = gm_mask.to(device)    # shape: (N, 1, z, f, T)
        
        # Create branch inputs:
        xGM = inputs * gm_mask             # Gray matter branch (non-GM voxels become 0)
        xWM = inputs * (1.0 - gm_mask)       # Complementary branch (white matter/other)
        
        # Forward pass through YNet (which now expects two inputs)
        preds = model(xGM, xWM)              # expected shape: (N, 2, z, f, T)
        
        # Optionally, apply the overall mask to focus loss on valid regions
        masked_predictions = preds * masks
        masked_targets     = targets * masks
        
        loss_curr = loss_fn(masked_predictions, masked_targets)
        
        optimizer.zero_grad()
        loss_curr.backward()
        optimizer.step()
        
        batch_size = inputs.size(0)
        epoch_loss += loss_curr.item() * batch_size
        num_samples += batch_size

    avg_loss = epoch_loss / num_samples if num_samples > 0 else 0.0
    return avg_loss

def validate_model_ynet(model, loss_fn, data_loader, device='cpu'):
    """
    Validation loop to compute the average loss for the YNet model.
    
    For each batch, splits the input into two branches using the provided
    gray matter mask: one branch for GM (input * gm_mask) and one for WM
    (input * (1 - gm_mask)). The model's predictions are then masked with the
    overall mask before computing the loss.
    
    Args:
        model (nn.Module): The YNet model with two encoders, callable as model(xGM, xWM).
        loss_fn (callable): Loss function (expects (predictions, targets)).
        data_loader (DataLoader): Yields batches of (dat_in, dat_out, mask, gm_mask).
            - dat_in: shape (N, 2, z, f, T)
            - dat_out: shape (N, 2, z, f, T)
            - mask: shape (N, 2, z, f, T) for valid regions.
            - gm_mask: shape (N, 1, z, f, T) indicating gray matter regions.
        device (str): 'cpu' or 'cuda'.
    
    Returns:
        float: The average loss over the validation dataset.
    """
    model.eval()
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for dat_in, dat_out, mask, gm_mask in data_loader:
            # Move tensors to the device
            dat_in  = dat_in.to(device)    # (N, 2, z, f, T)
            dat_out = dat_out.to(device)
            mask    = mask.to(device)
            gm_mask = gm_mask.to(device)    # (N, 1, z, f, T)
            
            # Prepare branch inputs:
            xGM = dat_in * gm_mask          # Gray matter branch
            xWM = dat_in * (1.0 - gm_mask)    # Complement branch
            
            # Forward pass through the model
            predictions = model(xGM, xWM)     # (N, 2, z, f, T)
            
            # If the model returns a tuple, extract the first element
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            
            # Apply the overall mask
            masked_predictions = predictions * mask
            masked_targets     = dat_out * mask
            
            loss_curr = loss_fn(masked_predictions, masked_targets)
            
            batch_size = dat_in.size(0)
            total_loss += loss_curr.item() * batch_size
            num_samples += batch_size

    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    return avg_loss


            




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

def undersample_FT_data_Non_Random(fourier_data, k, U):
    """
    Non-random undersampling of the 3D Fourier-transformed data in k-space.
    For k=1, undersamples every 2nd k_z index ([:,:,0,...] = 0, [:,:,1,...] unchanged, etc.).
    For k=2, performs the same but complementary ([:,:,0,...] unchanged, [:,:,1,...] = 0, etc.).

    Parameters
    ----------
    fourier_data : np.ndarray
        A numpy array of complex Fourier data with shape:
        (X, Y, Z, ...), where X, Y, Z are spatial dimensions, and ... indicates additional dimensions.

    k : int
        Determines the type of undersampling:
        - k=1: Undersample every 2nd k_z index.
        - k=2: Complementary undersampling (every 2nd k_z index, but complementary to k=1).

    U : float
        Unused in this function but kept for compatibility with other versions.

    Returns
    -------
    undersampled_data : np.ndarray
        A numpy array of the same shape as `fourier_data`, with appropriate undersampling applied.
    """
    if k not in [1, 2]:
        raise ValueError("This function only supports k=1 or k=2.")

    # Shape of the k-space
    shape = fourier_data.shape

    # Create an empty array for undersampled data
    undersampled_data = np.zeros_like(fourier_data)

    # Apply non-random undersampling
    
    if k == 1:
        # Apply even undersampling for every 2nd k_z index
        for z in range(shape[2]):  # Loop over k_z index
            if z % 2 == 1:  # Keep odd indices
                undersampled_data[:, :, z, ...] = fourier_data[:, :, z, ...]
                
    if k == 2: 
         for z in range(shape[2]):  # Loop over k_z index
            if z % 2 == 1:  # Keep odd indices
                undersampled_data[:, :, z, 0, ...] = fourier_data[:, :, z, 0, ...]
            if z % 2 == 0:  # Keep odd indices
                undersampled_data[:, :, z, 1, ...] = fourier_data[:, :, z, 1, ...]

    return undersampled_data


# def reshape_for_pytorch(data, k):
#     """
#     Reshapes complex-valued MRSI data into a format suitable for PyTorch CNN training.

#     Parameters:
#     - data: numpy array of shape (22, 22, 21, k, N), where:
#         * (22, 22) are spatial dimensions (image size),
#         * 21 is the number of slices along the z-axis,
#         * k is the number of features,
#         * N is the number of samples (data points).

#     Returns:
#     - reshaped_data: PyTorch tensor of shape (N, k * 2, 22, 22, 21).
#     """
#     # Separate real and imaginary parts
#     N = data.shape[-1]
#     real_part = np.real(data)
#     imag_part = np.imag(data)

#     # Interleave real and imaginary parts along the feature axis
#     if k == 1:
#         # Handle the special case for k = 1
#         interleaved = np.empty((22, 22, 21, 2, N), dtype=np.float32)
#         interleaved[:, :, :, 0, :] = real_part  # Real part
#         interleaved[:, :, :, 1, :] = imag_part  # Imaginary part
#     else:
#         # General case for k > 1
#         interleaved = np.empty((22, 22, 21, k * 2, N), dtype=np.float32)
#         interleaved[:, :, :, 0::2, :] = real_part  # Real parts at even indices
#         interleaved[:, :, :, 1::2, :] = imag_part  # Imaginary parts at odd indices

#     # Transpose to (N, k*2, 22, 22, 21) for PyTorch
#     reshaped_data = np.transpose(interleaved, (4, 3, 0, 1, 2))  # Shape: (N, k*2, 22, 22, 21)

#     return reshaped_data





def recover_original_format(network_output, spatial_dims, batch_dims):
    """
    Recover the original format of complex-valued data from the reshaped network output.

    Parameters
    ----------
    network_output : np.ndarray
        The network output of shape `(batch_size, 2, spatial_dims[0], spatial_dims[1])`, 
        where the second dimension contains real and imaginary components.
        
    spatial_dims : tuple
        The spatial dimensions of the original data, e.g., `(22, 22)`.
        
    batch_dims : tuple
        The batch dimensions of the original data, e.g., `(21, 96, 8)`.

    Returns
    -------
    recovered_data : np.ndarray
        The recovered complex-valued data of shape `(spatial_dims[0], spatial_dims[1], *batch_dims)`.
    """
    # Step 1: Reshape the flat batch dimension into the original batch dimensions
    recovered_data = network_output.reshape(*batch_dims, 2, *spatial_dims)  # Shape: (21, 96, 8, 2, 22, 22)

    # Step 2: Move the channel dimension (2) to the last axis for real and imaginary components
    recovered_data = np.transpose(recovered_data, (3, 4, 5, 0, 1, 2))  # Shape: (22, 22, 2, 21, 96, 8)

    # Step 3: Combine the real and imaginary parts into complex numbers
    recovered_data = recovered_data[0] + 1j * recovered_data[1]  # Shape: (22, 22, 21, 96, 8)

    return recovered_data






def reconstruct_image(tensor, dimensions): #this takes a vector in the format of the neural network input (k-space vector representation), and computes the actual real image through fourier transforming
    """
    Process a PyTorch tensor to perform the following operations:
    1. Reshape the tensor to (64, 64, 2)
    2. Convert to NumPy array
    3. Extract real and imaginary parts and combine them into a complex array
    4. Apply inverse FFT shift
    5. Apply the inverse 2D Fourier Transform
    6. Extract the real part of the result

    Parameters:
    tensor (torch.Tensor): Input tensor with shape (H, W, 2), where the last dimension represents real and imaginary parts.

    Returns:
    np.ndarray: The real part of the inverse 2D Fourier Transform.
    """
    # Reshape the tensor to (64, 64, 2) if not already in that shape
    tensor = tensor.reshape(dimensions, dimensions, 2)
    
    # Convert to NumPy array
    tensor_np = tensor.numpy()
    
    # Extract real and imaginary parts
    real_part = tensor_np[:, :, 0]
    imaginary_part = tensor_np[:, :, 1]
    
    # Combine into complex array
    complex_array = real_part + 1j * imaginary_part
    
    # Apply inverse FFT shift
    complex_array_unshifted = np.fft.ifftshift(complex_array)
    
    # Apply the inverse 2D Fourier Transform
    reconstructed_image = np.fft.ifft2(complex_array_unshifted)
    
    # Extract the real part of the result
    reconstructed_image_real = np.real(reconstructed_image)
    
    return reconstructed_image_real

def compute_relative_rmse(model, test_dataset, index):
    """
    Compute the relative RMSE for a given model and dataset example.

    Parameters:
    - model: The trained neural network model.
    - test_dataset: The dataset containing examples and their ground truths.
    - index: The index of the example in the dataset.

    Returns:
    - relative_rmse: The relative RMSE as a percentage.
    """

    # Extract the example from the dataset
    Example = test_dataset[index]

    # Model input and ground truth
    Model_Input = Example[0]
    OUTPUT, C2 = model(Model_Input)

    # Detach model output and ground truth for calculation
    model_output = OUTPUT.squeeze().detach()
    ground_truth = test_dataset[index][1]

    # Compute squared differences
    squared_diff = (model_output - ground_truth) ** 2

    # Compute Mean Squared Error (MSE)
    mse = torch.mean(squared_diff)

    # Compute root mean squared error (RMSE)
    rmse = torch.sqrt(mse)

    # Compute the mean of the ground truth values
    mean_ground_truth = torch.mean(ground_truth)

    # Compute relative RMSE (RRMSE) and express it as a percentage
    relative_rmse = (rmse / mean_ground_truth) * 100

    return relative_rmse.item()

def compute_rmse_distribution(model, dataset):
    """
    Compute the RMSE distribution for the entire validation dataset.

    Parameters:
    - model: The trained neural network model.
    - dataset: The validation dataset.

    Returns:
    - rmse_list: List of relative RMSE values for each example in the dataset.
    """
    rmse_list = []

    max_index = len(dataset)

    for index in range(max_index):
        relative_rmse = compute_relative_rmse(model, dataset, index)
        rmse_list.append(relative_rmse)

    return rmse_list


def rotate_image(image, angle):
    """
    Rotates a 2D image around its center by a given angle.

    Parameters:
        image (np.ndarray): The input 2D array representing an image.
        angle (float): The angle by which to rotate the image in degrees.

    Returns:
        np.ndarray: The rotated image.
    """
    rotated_image = rotate(image, angle, reshape=False)
    return rotated_image

# def group_time_steps(arr, k):
#     """
#     Group subsequent time steps in an array of shape (22,22,21, 8,N).
#     The fourth dimension (of size 8) corresponds to time steps.
#     We group them in chunks of size k.

#     Parameters
#     ----------
#     arr : np.ndarray
#         Input array of shape (22,22,21,8,N).
#     k : int
#         The grouping size for the time steps. Must divide 8.

#     Returns
#     -------
#     np.ndarray
#         If k=1, returns shape (22,22,21,8*N).
#         Otherwise, returns shape (22,22,21,k,(8/k)*N).
#     """
#     X, Y, Z, T, M = arr.shape
#     if T != 8:
#         raise ValueError("The time dimension must be 8.")
#     if 8 % k != 0:
#         raise ValueError("k must divide 8 evenly.")

#     # Special case: k=1 means just flatten the time dimension into the last dimension
#     if k == 1:
#         return arr.reshape(X, Y, Z, T * M)

#     # Move the time dimension (currently at index 2) to the end for easier grouping
#     # Original: (X,Y,Z,T,N) -> moveaxis T to the end -> (X,Y,Z,N,T)
#     arr = np.moveaxis(arr, 3, -1)  # arr.shape is now (22,22,21,N,8)

#     # Now we have arr with last dimension = 8 (time steps) and the third dimension = N.
#     # We want to group these 8 time steps into chunks of k.
#     # After grouping, we will have (8/k) groups. Each group becomes a 'k' dimension,
#     # and the number of 'groups' multiplies N by (8/k).

#     # Reshape so that the last two dimensions (N,8) become (N*(8/k), k)
#     arr = arr.reshape(X, Y, Z, M * (T // k), k)  # shape: (X,Y,(N*(8/k)),k)

#     # We want the final shape to be (X,Y,k,(8/k)*N), so we transpose the last two axes
#     arr = arr.transpose(0, 1, 2, 4, 3)  # shape: (X,Y,k,(N*(8/k)))

#     return arr

def normalize_data_per_image(input_data, ground_truth):
    """
    Normalizes input data and ground truth on a per-image basis to the range [0, 1]
    using the maximum absolute value of each image.

    Parameters:
    - input_data (numpy.ndarray): The input data array of shape (22, 22, 21, k, N) or (22, 22, 21, N).
    - ground_truth (numpy.ndarray): The ground truth data array with the same shape as input_data.

    Returns:
    - normalized_input (numpy.ndarray): Normalized input data.
    - normalized_ground_truth (numpy.ndarray): Normalized ground truth data.
    """
    # Ensure input_data and ground_truth have the same shape
    if input_data.shape != ground_truth.shape:
        raise ValueError("Input data and ground truth must have the same shape.")

    # Initialize normalized arrays with the same shape
    normalized_input = np.zeros_like(input_data)
    normalized_ground_truth = np.zeros_like(ground_truth)

    # Normalize each image independently
    if len(input_data.shape) == 5:  # 4D case (22, 22, 21, k, N)
        for k in range(input_data.shape[3]):
            for n in range(input_data.shape[4]):
                max_abs_value = np.max(np.abs(input_data[:, :, :, k, n]))
                if max_abs_value == 0:
                    raise ValueError(f"Maximum absolute value of the input data is zero for image k={k}, n={n}. Normalization is not possible.")
                normalized_input[:, :, :, k, n] = input_data[:, :, :, k, n] / max_abs_value
                normalized_ground_truth[:, :, :, k, n] = ground_truth[:, :, :, k, n] / max_abs_value

    elif len(input_data.shape) == 4:  # 3D case (22, 22, 21, N)
        for n in range(input_data.shape[3]):
            max_abs_value = np.max(np.abs(input_data[:, :, :, n]))
            if max_abs_value == 0:
                raise ValueError(f"Maximum absolute value of the input data is zero for image n={n}. Normalization is not possible.")
            normalized_input[:, :, :, n] = input_data[:, :, :, n] / max_abs_value
            normalized_ground_truth[:, :, :, n] = ground_truth[:, :, :, n] / max_abs_value

    else:
        raise ValueError("Unsupported data shape. Expected 4D or 5D input.")

    return normalized_input, normalized_ground_truth

# Example usage:
# Assuming `input_data` and `ground_truth` are NumPy arrays with shapes (22, 22, 21, k, N) or (22, 22, 21, N).
# normalized_input, normalized_ground_truth = normalize_data_per_image(input_data, ground_truth)


def split_spectrum_into_3(data, idxA=(0,55), idxB=(56,71), idxC=(72,None)):
    """
    data: torch.Tensor or np.ndarray, shape (N, 2, z, f, T)
    idxA, idxB, idxC: tuples specifying the frequency index range [start, end+1).
    
    Returns:
        xA, xB, xC: Tensors shaped (N, 2, z, f, T).
                    Each is zero except within its assigned freq range.
    """
    N, C, Z, F, T = data.shape
    
    # Convert to torch Tensor if it's not already
    if not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data)
    
    # Make clones or zeros
    xA = torch.zeros_like(data)
    xB = torch.zeros_like(data)
    xC = torch.zeros_like(data)
    
    # Define the ranges explicitly
    # e.g., idxA=(0,55) => freq indices [0..54]
    # if idxC=(72,None), that means from 72 to the end
    startA, endA = idxA
    startB, endB = idxB
    startC, endC = idxC  # if None, go to the end
    
    if endA is None: endA = F
    if endB is None: endB = F
    if endC is None: endC = F
    
    # Fill xA with data in [startA:endA]
    xA[:, :, :, startA:endA, :] = data[:, :, :, startA:endA, :]
    # Fill xB in [startB:endB]
    xB[:, :, :, startB:endB, :] = data[:, :, :, startB:endB, :]
    # Fill xC in [startC:endC]
    xC[:, :, :, startC:endC, :] = data[:, :, :, startC:endC, :]
    
    return xA, xB, xC


    
