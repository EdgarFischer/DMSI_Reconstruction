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

class RandomAugment3D:
    """
    Combined 3D augmentation:
      - (Optional) Random phase multiplication for complex data.
      - (Optional) Random 3D rotation by angles in x-y and x-z planes.
      - (Optional) Random shifting along x, y, z.
      - (Optional) Random 3D scaling along x, y, z axes.

    Assumes input tensors have shape (C, H, W, D).
    For complex data (real/imag pairs), C should be even.
    """

    def __init__(self,
                 rotation_range=0.3,
                 shift_pixels=1,
                 scale_range=0.1,
                 apply_phase=True,
                 apply_rotation=True,
                 apply_shift=True,
                 apply_scaling=True):
        """
        Args:
            rotation_range (float): Max absolute rotation (in radians) for each plane rotation.
                                   e.g., 0.3 ~ ±17° for both xy-plane and xz-plane.
            shift_pixels (int): Max shift in each spatial dimension (H, W, D).
            scale_range (float): Fractional scale range, e.g. 0.1 => scale factor in [0.9, 1.1].
            apply_phase (bool): Apply random phase to complex pairs.
            apply_rotation (bool): Apply random 3D rotation (xy + xz planes).
            apply_shift (bool): Apply random shift.
            apply_scaling (bool): Apply random scaling along x, y, z.
        """
        self.rotation_range = rotation_range
        self.shift_pixels = shift_pixels
        self.scale_range = scale_range
        self.apply_phase = apply_phase
        self.apply_rotation = apply_rotation
        self.apply_shift = apply_shift
        self.apply_scaling = apply_scaling

    # ------------------------------
    # PHASE AUGMENTATION
    # ------------------------------
    def apply_random_phase(self, tensor):
        """
        Expects an even number of channels (C). Each pair (real, imag) gets the same random phase shift.
        tensor shape: (C, H, W, D)
        """
        tensor = tensor.clone()  # avoid in-place modification
        C = tensor.shape[0]
        for i in range(0, C, 2):
            phi = random.uniform(-math.pi, math.pi)  # random phase in [-π, π]
            cos_phi = math.cos(phi)
            sin_phi = math.sin(phi)
            real = tensor[i]
            imag = tensor[i+1]
            # Multiply complex number by exp(i*phi)
            new_real = real * cos_phi - imag * sin_phi
            new_imag = real * sin_phi + imag * cos_phi
            tensor[i]   = new_real
            tensor[i+1] = new_imag
        return tensor

    # ------------------------------
    # 3D ROTATION
    # ------------------------------
    def apply_random_3d_rotation(self, data, label, mask):
        """
        Randomly rotate in the x-y plane (about z-axis) and x-z plane (about y-axis).
        Composes these two rotations into a single 3D affine transform, then applies
        it via grid_sample.
        """
        angle_xy = random.uniform(-self.rotation_range, self.rotation_range)  # rotation about z-axis
        angle_xz = random.uniform(-self.rotation_range, self.rotation_range)  # rotation about y-axis

        # Rotation about z-axis by angle_xy
        Rz = torch.tensor([
            [ math.cos(angle_xy), -math.sin(angle_xy), 0],
            [ math.sin(angle_xy),  math.cos(angle_xy), 0],
            [ 0,                  0,                  1]
        ], dtype=torch.float)

        # Rotation about y-axis by angle_xz
        Ry = torch.tensor([
            [ math.cos(angle_xz),  0, math.sin(angle_xz)],
            [ 0,                   1, 0               ],
            [-math.sin(angle_xz),  0, math.cos(angle_xz)]
        ], dtype=torch.float)

        # Combined rotation R = Ry * Rz (matrix multiplication)
        # Adjust order if you prefer Rz * Ry; 
        # here we do Rz first, then Ry.
        R = torch.matmul(Ry, Rz)

        # Build a 3x4 affine matrix (no translation).
        # [ R | t ], shape = (3, 4). PyTorch expects a float Tensor of shape (N=1, 3, 4).
        affine_3x4 = torch.zeros((3, 4), dtype=torch.float)
        affine_3x4[:, :3] = R  # put R into the left side
        # No translation => last column = 0

        # Turn this into (1, 3, 4) for affine_grid
        affine_3x4 = affine_3x4.unsqueeze(0)

        # Apply to each tensor
        def rotate_tensor(tensor_4d):
            # shape: (C, H, W, D) => (1, C, D, H, W)
            tensor_5d = tensor_4d.unsqueeze(0).permute(0, 1, 4, 2, 3)
            N, C, D, H, W = tensor_5d.shape
            grid = F.affine_grid(affine_3x4, size=(N, C, D, H, W), align_corners=True)
            tensor_rot = F.grid_sample(tensor_5d, grid,
                                       mode='bilinear',
                                       padding_mode='border',
                                       align_corners=True)
            # => (C, H, W, D)
            return tensor_rot.permute(0, 1, 3, 4, 2).squeeze(0)

        data_rot  = rotate_tensor(data)
        label_rot = rotate_tensor(label)
        mask_rot  = rotate_tensor(mask)
        return data_rot, label_rot, mask_rot

    # ------------------------------
    # SHIFT
    # ------------------------------
    def apply_random_shift(self, data, label, mask):
        """
        Shift the volume by up to ±self.shift_pixels along each dimension (H, W, D).
        """
        shift_h = random.randint(-self.shift_pixels, self.shift_pixels)
        shift_w = random.randint(-self.shift_pixels, self.shift_pixels)
        shift_d = random.randint(-self.shift_pixels, self.shift_pixels)

        data_shift  = torch.roll(data,  shifts=(shift_h, shift_w, shift_d), dims=(1, 2, 3))
        label_shift = torch.roll(label, shifts=(shift_h, shift_w, shift_d), dims=(1, 2, 3))
        mask_shift  = torch.roll(mask,  shifts=(shift_h, shift_w, shift_d), dims=(1, 2, 3))
        return data_shift, label_shift, mask_shift

    # ------------------------------
    # 3D SCALING
    # ------------------------------
    def apply_random_3d_scaling(self, data, label, mask):
        """
        Scale the volume in x, y, z directions by random scale factors in
        [1 - scale_range, 1 + scale_range].
        """
        scale_x = random.uniform(1 - self.scale_range, 1 + self.scale_range)
        scale_y = random.uniform(1 - self.scale_range, 1 + self.scale_range)
        scale_z = random.uniform(1 - self.scale_range, 1 + self.scale_range)

        # Construct diagonal scale matrix:
        S = torch.tensor([
            [scale_x, 0,       0      ],
            [0,       scale_y, 0      ],
            [0,       0,       scale_z]
        ], dtype=torch.float)

        affine_3x4 = torch.zeros((3, 4), dtype=torch.float)
        affine_3x4[:, :3] = S  # put scaling in the top-left 3x3
        # no translation => last column = 0
        affine_3x4 = affine_3x4.unsqueeze(0)  # (1, 3, 4)

        def scale_tensor(tensor_4d):
            # shape: (C, H, W, D) => (1, C, D, H, W)
            tensor_5d = tensor_4d.unsqueeze(0).permute(0, 1, 4, 2, 3)
            N, C, D, H, W = tensor_5d.shape
            grid = F.affine_grid(affine_3x4, size=(N, C, D, H, W), align_corners=True)
            tensor_scl = F.grid_sample(tensor_5d, grid,
                                       mode='bilinear',
                                       padding_mode='border',
                                       align_corners=True)
            return tensor_scl.permute(0, 1, 3, 4, 2).squeeze(0)

        data_scl  = scale_tensor(data)
        label_scl = scale_tensor(label)
        mask_scl  = scale_tensor(mask)
        return data_scl, label_scl, mask_scl

    # ------------------------------
    # COMBINED ENTRY POINT
    # ------------------------------
    def __call__(self, data, label, mask):
        """
        data, label, mask have shape (C, H, W, D).
        We apply (optionally) random phase, 3D rotation, shift, and scaling.
        """
        if self.apply_phase:
            data  = self.apply_random_phase(data)
            label = self.apply_random_phase(label)

        if self.apply_rotation:
            data, label, mask = self.apply_random_3d_rotation(data, label, mask)

        if self.apply_shift:
            data, label, mask = self.apply_random_shift(data, label, mask)

        if self.apply_scaling:
            data, label, mask = self.apply_random_3d_scaling(data, label, mask)

        return data, label, mask


# Example integration into a custom dataset:
class TensorDatasetWithAugmentation(torch.utils.data.Dataset):
    def __init__(self, data, labels, masks, norm_values, transform=None):
        """
        Args:
            data: numpy array of shape (N, 2*grouped_time_steps, 22, 22, 21)
            labels: numpy array of shape (N, 2*grouped_time_steps, 22, 22, 21)
            masks: numpy array of shape (N, 2*grouped_time_steps, 22, 22, 21)
            norm_values: list or numpy array of length N with a normalization value for each data entry
            transform: callable, applied as transform(data, label, mask, norm_value)
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.float32)
        
        # Convert norm_values to a tensor if it's a list or numpy array.
        if isinstance(norm_values, (list, np.ndarray)):
            self.norm_values = torch.tensor(norm_values, dtype=torch.float32)
        else:
            self.norm_values = norm_values
        
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]     # Shape: (C, H, W, D)
        label = self.labels[idx]   # Shape: (C, H, W, D)
        mask = self.masks[idx]     # Shape: (C, H, W, D)
        norm_value = self.norm_values[idx]  # Normalization value for this sample
        
#         if self.transform:
#             data, label, mask = self.transform(data, label, mask)
            
        return data, label, mask, norm_value
    
    
###### STOP AUGMENTATION


class TensorDataset(Dataset):
    def __init__(self, data, labels, masks):
        """
        Initializes the dataset.

        Parameters:
        - data: numpy array of shape (N, 2*grouped_time_steps, 22, 22, 21) for input data.
        - labels: numpy array of shape (N, 2*grouped_time_steps, 22, 22, 21) for ground truth labels.
        - masks: numpy array of shape (N, 2, 22, 22, 21) with 0s and 1s, used to mask out noise.
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.float32)  # Alternatively, use dtype=torch.bool

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]     # Shape: (2*grouped_time_steps, 22, 22, 21)
        label = self.labels[idx]   # Shape: (2*grouped_time_steps, 22, 22, 21)
        mask = self.masks[idx]     # Shape: (2, 22, 22, 21)
        return data, label, mask
    
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

class UNet3D(nn.Module):
    """
    A 3D U-Net for image enhancement, without any masking logic.
    Includes a residual addition of the original input.
    """
    def __init__(self, grouped_time_steps=1, use_batch_norm=False):
        """
        Args:
            grouped_time_steps (int): Number of stacked time steps (k).
            use_batch_norm (bool): Whether to use batch normalization.
        """
        super(UNet3D, self).__init__()
        
        self.k = grouped_time_steps
        self.use_batch_norm = use_batch_norm
        
        # Number of channels in the input (2 per time step: real + imaginary)
        in_channels = grouped_time_steps * 2
        # Number of channels in the output (same as input)
        out_channels = 1
        
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
        
        # Final 3D conv to get back to "out_channels"
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)

    

    def forward(self, dat):
        """
        Forward pass of the 3D U-Net, ignoring any masking logic.

        Args:
            dat (torch.Tensor): Input of shape (batch_size, 2*k, H, W, D).

        Returns:
            tuple(torch.Tensor, torch.Tensor): (output, C2)
                - output is the final enhanced volume with shape (batch_size, 1, H, new_W, D)
                - C2 is an intermediate feature map.
        """
        # ---------------- Encoder ----------------
        e1 = self.enc1(dat)       # (batch, 64, 24, 96, 8)
        p1 = self.pool(e1)        # downsample

        e2 = self.enc2(p1)        # (batch, 128, 12, 48, 4)
        p2 = self.pool(e2)        # downsample

        e3 = self.enc3(p2)        # (batch, 256, 6, 24, 2)
        p3 = self.pool(e3)        # downsample

        # ---------------- Bottleneck ----------------
        b = self.bottleneck(p3)   # (batch, 512, 3, 12, 1)

        # ---------------- Decoder ----------------
        u3 = self.up3(b)               # upsample -> (batch, 256, 6, 24, 2)
        cat3 = torch.cat([u3, e3], 1)   # skip connection
        d3 = self.dec3(cat3)           # (batch, 256, 6, 24, 2)

        u2 = self.up2(d3)              # upsample -> (batch, 128, 12, 48, 4)
        cat2 = torch.cat([u2, e2], 1)   # skip connection
        d2 = self.dec2(cat2)           # (batch, 128, 12, 48, 4)

        u1 = self.up1(d2)              # upsample -> (batch, 64, 24, 96, 8)
        cat1 = torch.cat([u1, e1], 1)   # skip connection
        C2 = self.dec1(cat1)           # (batch, 64, 24, 96, 8)

        # ---------------- Final Conv ----------------
        x = self.final_conv(C2)        # (batch, 1, 24, 96, 8)

        # ------------ Upsample Width Only ------------
        # Upsample the width dimension from 96 to 162 while keeping height and depth.
        x = F.interpolate(x, size=(x.size(2), 162, x.size(4)), mode='trilinear', align_corners=False)

        return x, C2




class CustomLoss(torch.nn.Module):    # the purpose of this class is to define a custo loss function with L1 penalty as at the C2 layer activation, as described in the paper
    def __init__(self, l1_lambda=0.000):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_lambda = l1_lambda

    def forward(self, outputs, targets, model, C2):
        mse_loss = self.mse_loss(outputs, targets)
        l1_loss = self.l1_lambda * torch.sum(torch.abs(C2))
        total_loss = mse_loss + l1_loss
        return total_loss


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

def train_one_epoch(model, optimizer, loss_fn, data_loader, device='cpu'):
    model.train()  # Set model to training mode
    
    total_loss = 0.0
    num_samples = 0
    
    for dat_in, dat_out, mask in data_loader:
        # Move input, output, and mask to the device
        dat_in, dat_out, mask = dat_in.to(device), dat_out.to(device), mask.to(device)

        optimizer.zero_grad()  # Clear previous gradients
        
        predictions, C2 = model(dat_in)  # Forward pass
        
        # --- CROP TO ORIGINAL SIZE (22x22x21) ---
        predictions_cropped = predictions#[:, :, 1:-1, 1:-1, 1:-2]  # (N, C, 22, 22, 21)
        targets_cropped = dat_out#[:, :, 1:-1, 1:-1, 1:-2]          # (N, C, 22, 22, 21)
        mask_cropped = mask#[:, :, 1:-1, 1:-1, 1:-2]                # (N, C, 22, 22, 21)

        # Compute loss only on the cropped region
        masked_predictions = predictions_cropped * mask_cropped
        masked_targets = targets_cropped * mask_cropped
        loss_curr = loss_fn(masked_predictions, masked_targets, model, C2)
        
        loss_curr.backward()  # Compute gradients
        optimizer.step()  # Update model weights
        
        total_loss += loss_curr.item() * dat_in.size(0)  # Accumulate loss for this batch
        num_samples += dat_in.size(0)  # Count samples in this batch
        
    # Return the average loss for the epoch
    return total_loss / num_samples

        

import torch

def validate_model(model, loss_fn, data_loader, device='cpu'):
    """
    Validation loop to compute average loss using masks to focus on valid regions.
    """
    model.eval()  # Set model to evaluation mode

    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for dat_in, dat_out, mask in data_loader:
            # Move inputs, outputs, and mask to the specified device
            dat_in  = dat_in.to(device)
            dat_out = dat_out.to(device)
            mask    = mask.to(device)
            
            # Forward pass
            predictions, C2 = model(dat_in)

            # --- Crop to original size if necessary ---
            predictions_cropped = predictions  # Adjust cropping if required
            targets_cropped     = dat_out
            mask_cropped        = mask
            
            # Compute masked outputs
            masked_predictions = predictions_cropped * mask_cropped
            masked_targets     = targets_cropped * mask_cropped

            # Compute loss on the cropped (and masked) data
            loss_curr = loss_fn(masked_predictions, masked_targets, model, C2)
            
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





    
