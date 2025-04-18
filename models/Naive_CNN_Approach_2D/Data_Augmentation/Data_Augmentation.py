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

import numpy as np
from scipy.ndimage import rotate, shift, zoom

import numpy as np
from scipy.ndimage import rotate, shift, zoom

def crop_or_pad(array, target_shape):
    """
    Crops or pads an array to match the target shape.
    
    Parameters:
        array (numpy.ndarray): Input 2D array.
        target_shape (tuple): Target shape (height, width).
    
    Returns:
        numpy.ndarray: Resized array with the target shape.
    """
    target_h, target_w = target_shape
    h, w = array.shape
    
    # Crop if larger
    if h > target_h:
        crop_h = (h - target_h) // 2
        array = array[crop_h:crop_h + target_h, :]
    if w > target_w:
        crop_w = (w - target_w) // 2
        array = array[:, crop_w:crop_w + target_w]
    
    # Pad if smaller
    pad_h = max(0, (target_h - array.shape[0]) // 2)
    pad_w = max(0, (target_w - array.shape[1]) // 2)
    array = np.pad(array, ((pad_h, target_h - array.shape[0] - pad_h),
                           (pad_w, target_w - array.shape[1] - pad_w)),
                   mode='constant')
    
    return array

def augment_multiple(data, num_augmentations=3, scale_range=(0.95, 1.05), shift_range=(-1, 1), rotation_range=(-0.05, 0.05)):
    H, W, T, N = data.shape
    augmented_data = np.zeros((H, W, T, N * num_augmentations), dtype=data.dtype)
    
    index = 0
    for i in range(N):  # Iterate over datasets
        for _ in range(num_augmentations):  # Create multiple augmentations per dataset
            # Generate random transformations
            scale_factor = np.random.uniform(scale_range[0], scale_range[1])
            shift_values = np.random.uniform(shift_range[0], shift_range[1], size=2)  # (dx, dy)
            rotation_angle = np.random.uniform(rotation_range[0], rotation_range[1]) * (180 / np.pi)  # Convert to degrees
            
            for t in range(T):  # Apply the same transformation to all temporal slices
                real_part = data[:, :, t, i].real
                imag_part = data[:, :, t, i].imag
                
                real_aug = zoom(real_part, scale_factor, order=1)
                real_aug = shift(real_aug, shift_values, order=1)
                real_aug = rotate(real_aug, rotation_angle, reshape=False, order=1)
                
                imag_aug = zoom(imag_part, scale_factor, order=1)
                imag_aug = shift(imag_aug, shift_values, order=1)
                imag_aug = rotate(imag_aug, rotation_angle, reshape=False, order=1)
                
                real_aug = crop_or_pad(real_aug, (H, W))
                imag_aug = crop_or_pad(imag_aug, (H, W))
                
                augmented_data[:, :, t, index] = real_aug + 1j * imag_aug
            index += 1
    
    return augmented_data
