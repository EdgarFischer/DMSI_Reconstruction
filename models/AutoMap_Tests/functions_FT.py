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

class TensorDataset(Dataset):
    def __init__(self, data, labels):
        # convert numpy arrays to Pytorch tensors
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx].permute(2, 0, 1)  # Add an extra dimension to match the model output shape
        return data, label
    
def generate_model(n):
    class Net_ver_0_0(torch.nn.Module):
        def __init__(self, n):  # note that n is the value mentioned in the paper. The images are of size nxn
            super(Net_ver_0_0, self).__init__()
            
            Nin_fc = 2 * (n * n)  # Input layer nodes
            Nhid = (n * n) # hidden layer nodes
            
            # define fully connected layers first
            self.fc1 = nn.Linear(2 * n * n, n * n)
            self.fc2 = nn.Linear(n * n, n * n)
            self.fc3 = nn.Linear(n * n, n * n)
            
            self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
            self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
            self.deconv = nn.ConvTranspose2d(64, 2, kernel_size=7, stride=1, padding=3)
            self.relu = nn.LeakyReLU()
            self.tanh = nn.Tanh()
            
            # Define convolutional layers
            #self.conv_layers = nn.Sequential
            #    nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2), nn.LeakyReLU(), # C1 note that padding has to be chosen as 2, to ensure that the output matrices have again dimensions nxn
            #    nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2), nn.LeakyReLU(),# C2
            #    nn.ConvTranspose2d(64, 1, kernel_size=7, stride=1, padding=3)    # Deconvolution layer, padding 3 ensures again correct output dimensions
            
            
        def forward(self, dat):
            x = self.relu(self.fc1(dat))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            
            # Reshape to n x n matrix for convolutional layers
            x = x.view(-1, 1, n, n)  # Assuming single-channel input for Conv2d
            x = self.relu(self.conv1(x))
            C2 = self.relu(self.conv2(x)) # this is also returned, to induce a penalty for large C2 activations, as described in the paper
            x = self.deconv(C2)
            
            return x, C2

    model = Net_ver_0_0(n)

    return model

class CustomLoss(torch.nn.Module):    # the purpose of this class is to define a custo loss function with L1 penalty as at the C2 layer activation, as described in the paper
    def __init__(self, l1_lambda=0.0001):
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
      

def train_one_epoch(model, optimizer, loss_fn, data_loader, device='cpu'):
    model.train()  # Set model to training mode
    
    total_loss = 0.0
    num_samples = 0
    
    for dat_in, dat_out in data_loader:
        dat_in, dat_out = dat_in.to(device), dat_out.to(device)  # Move to device
        
        optimizer.zero_grad()  # Clear previous gradients
        predictions, C2 = model(dat_in)  # Forward pass
        loss_curr = loss_fn(predictions, dat_out, model, C2)  # Compute loss
        
        loss_curr.backward()  # Compute gradients
        optimizer.step()  # Update model weights
        
        total_loss += loss_curr.item() * dat_in.size(0)  # Accumulate loss for this batch
        num_samples += dat_in.size(0)  # Count samples in this batch
        
    # Return the average loss for the dataset
    return total_loss / num_samples
        

def validate_model(model, loss_fn, data_loader, device='cpu'):
    model.eval()  # Set model to evaluation mode
    
    total_loss = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for dat_in, dat_out in data_loader:
            dat_in, dat_out = dat_in.to(device), dat_out.to(device)  # Move to device
            predictions, C2 = model(dat_in)  # Forward pass
            loss_curr = loss_fn(predictions, dat_out, model, C2) # Compute loss
            total_loss += loss_curr.item() * dat_in.size(0)  # Accumulate loss
            num_samples += dat_in.size(0)  # Count samples
    
    # Return the average loss for the dataset
    return total_loss / num_samples

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
    f_transform = np.fft.fft2(images, axes=(0, 1))
    
    # Shift the zero frequency component to the center
    f_transform_shifted = np.fft.fftshift(f_transform, axes=(0, 1))
    
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
    shifted_data = np.fft.ifftshift(fourier_data, axes=(0, 1))
    
    # Perform the 2D Inverse FFT along the first two dimensions
    inverse_transform = np.fft.ifft2(shifted_data, axes=(0, 1))
    
    return inverse_transform

def undersample_FT_data(fourier_data):
    """
    Undersamples the Fourier-transformed data by setting every third row in the first 
    dimension to zero.
    
    Parameters
    ----------
    fourier_data : np.ndarray
        A 5D numpy array of complex Fourier data with shape (N1, N2, N3, N4, N5).
        
    Returns
    -------
    undersampled_data : np.ndarray
        A 5D numpy array of the same shape as `fourier_data`, with every third row
        in the first dimension set to zero.
    """
    # Create a copy of the input array to avoid modifying the original
    undersampled_data = fourier_data.copy()
    
    # Loop through the first two dimensions and apply the mask
    undersampled_data[2::3, :, ...] = 0  # Set every third row to zero along axis 0
    
    return undersampled_data

import numpy as np

def reshape_data(fourier_data, other_data):
    """
    Generalized function to reshape two complex-valued arrays into real-valued arrays 
    for neural network input, with dimensions extracted automatically.
    
    Parameters
    ----------
    fourier_data : np.ndarray
        A complex-valued numpy array with any shape, where the first two dimensions 
        correspond to spatial dimensions.
        
    other_data : np.ndarray
        A complex-valued numpy array with any shape, where the first two dimensions 
        correspond to spatial dimensions.
        
    Returns
    -------
    reshaped_fourier_data : np.ndarray
        A real-valued numpy array of shape ((product of batch dimensions), (spatial dimensions * 2)).
        
    reshaped_other_data : np.ndarray
        A real-valued numpy array of shape ((product of batch dimensions), spatial dimension 0, spatial dimension 1, 2).
    """
    # Extract the spatial dimensions and batch dimensions from the Fourier data
    spatial_dims_fourier = fourier_data.shape[:2]
    batch_dims_fourier = fourier_data.shape[2:]
    
    # Extract the spatial dimensions and batch dimensions from the other data
    spatial_dims_other = other_data.shape[:2]
    batch_dims_other = other_data.shape[2:]
    
    # Ensure spatial dimensions match between the two arrays
    if spatial_dims_fourier != spatial_dims_other:
        raise ValueError("Spatial dimensions of the two inputs do not match.")
    
    # For Fourier data (k-space):
    # Step 1: Separate real and imaginary parts
    real_part_fourier = np.real(fourier_data)
    imag_part_fourier = np.imag(fourier_data)
    real_imag_fourier = np.stack([real_part_fourier, imag_part_fourier], axis=-1)  # Shape: (N1, N2, ..., 2)
    
    # Step 2: Reshape batch dimensions into one dimension while preserving spatial dims
    batch_size_fourier = np.prod(batch_dims_fourier)
    reshaped_fourier_data = real_imag_fourier.reshape(spatial_dims_fourier[0], spatial_dims_fourier[1], batch_size_fourier, 2)
    
    # Step 3: Transpose to move batch dimension to the front
    reshaped_fourier_data = np.transpose(reshaped_fourier_data, (2, 0, 1, 3))  # Shape: (batch_size, N1, N2, 2)
    
    # Flatten spatial dimensions for NN input
    reshaped_fourier_data = reshaped_fourier_data.reshape(batch_size_fourier, np.prod(spatial_dims_fourier) * 2)  # Shape: (batch_size, spatial_dims * 2)
    
    # For other data (image):
    # Step 1: Separate real and imaginary parts
    real_part_other = np.real(other_data)
    imag_part_other = np.imag(other_data)
    real_imag_other = np.stack([real_part_other, imag_part_other], axis=-1)  # Shape: (N1, N2, ..., 2)
    
    # Step 2: Reshape batch dimensions into one dimension while preserving spatial dims
    batch_size_other = np.prod(batch_dims_other)
    reshaped_other_data = real_imag_other.reshape(spatial_dims_other[0], spatial_dims_other[1], batch_size_other, 2)  # Shape: (N1, N2, batch_size_other, 2)
    
    # Step 3: Transpose to move batch dimension to the front
    reshaped_other_data = np.transpose(reshaped_other_data, (2, 0, 1, 3))  # Shape: (batch_size, N1, N2, 2)
    
    return reshaped_fourier_data, reshaped_other_data


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
    
