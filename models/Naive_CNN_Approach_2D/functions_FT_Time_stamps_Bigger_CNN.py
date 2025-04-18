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
        data = self.data[idx].permute(2, 0, 1) 
        label = self.labels[idx].permute(2, 0, 1)  # Add an extra dimension to match the model output shape
        return data, label
    
def generate_model(n, k=1):
    class Net_ver_0_2(nn.Module):
        def __init__(self, n, k):
            super(Net_ver_0_2, self).__init__()
            
            # Convolutional layers for image enhancement
            self.conv1 = nn.Conv2d(k * 2, 64, kernel_size=5, stride=1, padding=2)
            self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)  # New layer
            
            self.deconv = nn.ConvTranspose2d(64, 2 * k, kernel_size=7, stride=1, padding=3)
            
            # Activation functions
            self.relu = nn.LeakyReLU()
            self.tanh = nn.Tanh()  # Use if output needs to be scaled between -1 and 1
            
        def forward(self, dat):
            # dat shape: (batch_size, 2*k, n, n)
            x = self.relu(self.conv1(dat))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))  # Pass through new layer
            C2 = x  # Optional: Retain C2 for regularization if needed
            x = self.deconv(C2)  # Produces (batch_size, 2*k, n, n)
            
            # Optionally apply final activation function (e.g., tanh)
            # x = self.tanh(x)
            return x, C2

    model = Net_ver_0_2(n, k)
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
    Undersamples the Fourier-transformed data conditionally based on time steps.

    On odd time steps (1, 3, 5, ...), every third row in the first dimension (X) is set to zero 
    starting from index 2. On even time steps (0, 2, 4, ...), every third row is set to zero 
    starting from index 0.

    The input data is assumed to have at least three dimensions: 
    The first two dimensions (X, Y) are spatial, the third dimension (T) is time, 
    and any number of trailing dimensions can follow.

    Parameters
    ----------
    fourier_data : np.ndarray
        A numpy array of complex Fourier data with shape:
        (X, Y, T, ...) 
        where X and Y are spatial dimensions, T is time, and ... indicates zero or more additional dimensions.

    Returns
    -------
    undersampled_data : np.ndarray
        A numpy array of the same shape as `fourier_data`, undersampled as described.
    """
    undersampled_data = fourier_data.copy()

    # Extract the dimensions
    X, Y, T = undersampled_data.shape[:3]
    # The rest of the dimensions (if any) follow after T

    for t in range(T):
        if t % 2 == 1:
            # Odd time steps: zero out every third row starting from index 2
            undersampled_data[2::3, :, t, ...] = 0
        else:
            # Even time steps: zero out every third row starting from index 0
            undersampled_data[::3, :, t, ...] = 0

    return undersampled_data


def reshape_for_pytorch(data, k):
    """
    Reshapes complex-valued MRSI data into a format suitable for PyTorch CNN training.

    Parameters:
    - data: numpy array of shape (22, 22, k, N), where:
        * (22, 22) are spatial dimensions (image size),
        * k is the number of features,
        * N is the number of samples (data points).
    
    Returns:
    - reshaped_data: PyTorch tensor of shape (N, k * 2, 22, 22).
    """
    # Separate real and imaginary parts
    N = data.shape[-1]
    real_part = np.real(data)
    imag_part = np.imag(data)
    
        # Interleave real and imaginary parts along the feature axis
    if k == 1:
        # Handle the special case for k = 1
        interleaved = np.empty((22, 22, 2, N), dtype=np.float32)
        interleaved[:, :, 0, :] = real_part # Real part
        interleaved[:, :, 1, :] = imag_part  # Imaginary part
    else:
        # General case for k > 1
        interleaved = np.empty((22, 22, k * 2, N), dtype=np.float32)
        interleaved[:, :, 0::2, :] = real_part  # Real parts at even indices
        interleaved[:, :, 1::2, :] = imag_part  # Imaginary parts at odd indices
    
    # Transpose to (N, k*2, 22, 22) for PyTorch
    reshaped_data = np.transpose(interleaved, (3, 0, 1, 2))  # Shape: (N, k*2, 22, 22)
    
    return reshaped_data





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

def group_time_steps(arr, k):
    """
    Group subsequent time steps in an array of shape (22,22,8,N).
    The third dimension (of size 8) corresponds to time steps.
    We group them in chunks of size k.

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (22,22,8,N).
    k : int
        The grouping size for the time steps. Must divide 8.

    Returns
    -------
    np.ndarray
        If k=1, returns shape (22,22,8*N).
        Otherwise, returns shape (22,22,k,(8/k)*N).
    """
    X, Y, T, M = arr.shape
    if T != 8:
        raise ValueError("The time dimension must be 8.")
    if 8 % k != 0:
        raise ValueError("k must divide 8 evenly.")

    # Special case: k=1 means just flatten the time dimension into the last dimension
    if k == 1:
        return arr.reshape(X, Y, T * M)

    # Move the time dimension (currently at index 2) to the end for easier grouping
    # Original: (X,Y,T,N) -> moveaxis T to the end -> (X,Y,N,T)
    arr = np.moveaxis(arr, 2, -1)  # arr.shape is now (22,22,N,8)

    # Now we have arr with last dimension = 8 (time steps) and the third dimension = N.
    # We want to group these 8 time steps into chunks of k.
    # After grouping, we will have (8/k) groups. Each group becomes a 'k' dimension,
    # and the number of 'groups' multiplies N by (8/k).

    # Reshape so that the last two dimensions (N,8) become (N*(8/k), k)
    arr = arr.reshape(X, Y, M * (T // k), k)  # shape: (X,Y,(N*(8/k)),k)

    # We want the final shape to be (X,Y,k,(8/k)*N), so we transpose the last two axes
    arr = arr.transpose(0, 1, 3, 2)  # shape: (X,Y,k,(N*(8/k)))

    return arr





    
