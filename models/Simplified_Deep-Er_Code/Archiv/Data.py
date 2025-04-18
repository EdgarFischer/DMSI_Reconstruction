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
    def __init__(self, k_space, image_reconstructed, ground_truth):
        """
        Initializes the dataset with reshaped data.

        Parameters:
        - k_space: Undersampled k-space data, shape (N, 2, 16, 16, 21).
        - image_reconstructed: Reconstructed images from k-space, shape (N, 2, 16, 16, 21).
        - ground_truth: Fully sampled ground truth images, shape (N, 2, 16, 16, 21).
        """
        self.k_space = torch.tensor(k_space, dtype=torch.float32)
        self.image_reconstructed = torch.tensor(image_reconstructed, dtype=torch.float32)
        self.ground_truth = torch.tensor(ground_truth, dtype=torch.float32)

    def __len__(self):
        return len(self.ground_truth)

    def __getitem__(self, idx):
        """
        Fetches the inputs and ground truth for a given index.

        Returns:
        - inputs: Tuple of (img_in, freq_in, inputs_img, inputs_kspace).
        - ground_truth: Fully sampled ground truth image.
        """
        # Prepare the input tuple
        img_in = self.image_reconstructed[idx]  # Reconstructed image input
        freq_in = self.k_space[idx]  # Undersampled k-space input
        inputs_img = self.image_reconstructed[idx]  # Same as img_in
        inputs_kspace = self.k_space[idx]  # Same as freq_in

        inputs = (inputs_img, inputs_kspace)
        ground_truth = self.ground_truth[idx]

        return inputs, ground_truth

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


def reshape_for_pytorch(data, k):
    """
    Reshapes complex-valued MRSI data into a format suitable for PyTorch CNN training.

    Parameters:
    - data: numpy array of shape (22, 22, 21, k, N), where:
        * (22, 22) are spatial dimensions (image size),
        * 21 is the number of slices along the z-axis,
        * k is the number of features,
        * N is the number of samples (data points).

    Returns:
    - reshaped_data: PyTorch tensor of shape (N, k * 2, 22, 22, 21).
    """
    # Separate real and imaginary parts
    N = data.shape[-1]
    real_part = np.real(data)
    imag_part = np.imag(data)

    # Interleave real and imaginary parts along the feature axis
    if k == 1:
        # Handle the special case for k = 1
        interleaved = np.empty((22, 22, 21, 2, N), dtype=np.float32)
        interleaved[:, :, :, 0, :] = real_part  # Real part
        interleaved[:, :, :, 1, :] = imag_part  # Imaginary part
    else:
        # General case for k > 1
        interleaved = np.empty((22, 22, 21, k * 2, N), dtype=np.float32)
        interleaved[:, :, :, 0::2, :] = real_part  # Real parts at even indices
        interleaved[:, :, :, 1::2, :] = imag_part  # Imaginary parts at odd indices

    # Transpose to (N, k*2, 22, 22, 21) for PyTorch
    reshaped_data = np.transpose(interleaved, (4, 3, 0, 1, 2))  # Shape: (N, k*2, 22, 22, 21)

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

def group_time_steps(arr, k):
    """
    Group subsequent time steps in an array of shape (22,22,21, 8,N).
    The fourth dimension (of size 8) corresponds to time steps.
    We group them in chunks of size k.

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (22,22,21,8,N).
    k : int
        The grouping size for the time steps. Must divide 8.

    Returns
    -------
    np.ndarray
        If k=1, returns shape (22,22,21,8*N).
        Otherwise, returns shape (22,22,21,k,(8/k)*N).
    """
    X, Y, Z, T, M = arr.shape
    if T != 8:
        raise ValueError("The time dimension must be 8.")
    if 8 % k != 0:
        raise ValueError("k must divide 8 evenly.")

    # Special case: k=1 means just flatten the time dimension into the last dimension
    if k == 1:
        return arr.reshape(X, Y, Z, T * M)

    # Move the time dimension (currently at index 2) to the end for easier grouping
    # Original: (X,Y,Z,T,N) -> moveaxis T to the end -> (X,Y,Z,N,T)
    arr = np.moveaxis(arr, 3, -1)  # arr.shape is now (22,22,21,N,8)

    # Now we have arr with last dimension = 8 (time steps) and the third dimension = N.
    # We want to group these 8 time steps into chunks of k.
    # After grouping, we will have (8/k) groups. Each group becomes a 'k' dimension,
    # and the number of 'groups' multiplies N by (8/k).

    # Reshape so that the last two dimensions (N,8) become (N*(8/k), k)
    arr = arr.reshape(X, Y, Z, M * (T // k), k)  # shape: (X,Y,(N*(8/k)),k)

    # We want the final shape to be (X,Y,k,(8/k)*N), so we transpose the last two axes
    arr = arr.transpose(0, 1, 2, 4, 3)  # shape: (X,Y,k,(N*(8/k)))

    return arr

def normalize_data_per_image(input_data, input_data_k_space, ground_truth):
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
    normalized_k_space = np.zeros_like(input_data)
    normalized_ground_truth = np.zeros_like(ground_truth)

    # Normalize each image independently
    if len(input_data.shape) == 5:  # 4D case (22, 22, 21, k, N)
        max_abs_values = np.zeros((input_data.shape[3], input_data.shape[4]))  # Shape (k, N)
        for k in range(input_data.shape[3]):
            for n in range(input_data.shape[4]):
                max_abs_value = np.max(np.abs(input_data[:, :, :, k, n]))
                if max_abs_value == 0:
                    raise ValueError(f"Maximum absolute value of the input data is zero for image k={k}, n={n}. Normalization is not possible.")
                max_abs_values[k, n] = max_abs_value
                normalized_input[:, :, :, k, n] = input_data[:, :, :, k, n] / max_abs_value
                normalized_ground_truth[:, :, :, k, n] = ground_truth[:, :, :, k, n] / max_abs_value
                normalized_k_space[:, :, :, k, n] = normalized_k_space[:, :, :, k, n] / max_abs_value

    elif len(input_data.shape) == 4:  # 3D case (22, 22, 21, N)
        max_abs_values = np.zeros(input_data.shape[3])  # Shape (N)
        
        for n in range(input_data.shape[3]):
            max_abs_value = np.max(np.abs(input_data[:, :, :, n]))
            if max_abs_value == 0:
                raise ValueError(f"Maximum absolute value of the input data is zero for image n={n}. Normalization is not possible.")
            max_abs_values[n] = max_abs_value
            normalized_input[:, :, :, n] = input_data[:, :, :, n] / max_abs_value
            normalized_ground_truth[:, :, :, n] = ground_truth[:, :, :, n] / max_abs_value
            normalized_k_space[:, :, :, n] = normalized_k_space[:, :, :, n] / max_abs_value
    else:
        raise ValueError("Unsupported data shape. Expected 4D or 5D input.")

    return normalized_input, normalized_k_space, normalized_ground_truth, max_abs_values

# Example usage:
# Assuming `input_data` and `ground_truth` are NumPy arrays with shapes (22, 22, 21, k, N) or (22, 22, 21, N).
# normalized_input, normalized_ground_truth = normalize_data_per_image(input_data, ground_truth)





    
