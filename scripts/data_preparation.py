from scipy.io import loadmat
import numpy as np
from data_undersampling import apply_undersampling
import torch

#### data preparation ####
  
def reshape_for_pytorch(data, grouped_time_steps, is_mask=False):
    """
    Reshapes input data into a format suitable for PyTorch CNN training.
    
    For complex-valued MRSI data (is_mask=False), the input is expected to have shape:
        (x, y, z, grouped_time_steps, N)
    and the output tensor will have shape:
        (N, grouped_time_steps * 2, x, y, z)
    (i.e. interleaved real and imaginary channels).
    
    For mask data (is_mask=True), the input is assumed to be real-valued and have shape:
        (x, y, z, N)
    and the output tensor will have shape:
        (N, 1, x, y, z)
    
    Parameters:
        data (numpy.ndarray): The input array.
        grouped_time_steps (int): Number of time steps grouped (only relevant if is_mask=False).
        is_mask (bool): If True, treat the input as a real-valued mask and output a single channel.
    
    Returns:
        numpy.ndarray: The reshaped data as a float32 array.
    """
    if is_mask:
        # For masks, assume data is real and has shape (x, y, z, N)
        # We want to output shape (N, 1, x, y, z)
        reshaped_data = np.transpose(data, (3, 0, 1, 2))  # (N, x, y, z)
        reshaped_data = np.expand_dims(reshaped_data, axis=1)  # (N, 1, x, y, z)
        return reshaped_data.astype(np.float32)
    else:
        # For complex data: input shape is (x, y, z, grouped_time_steps, N)
        x, y, z = data.shape[0], data.shape[1], data.shape[2]
        N = data.shape[-1]
        real_part = np.real(data)
        imag_part = np.imag(data)
    
        if grouped_time_steps == 1:
            # Special case: grouped_time_steps == 1, create shape (x, y, z, 2, N)
            interleaved = np.empty((x, y, z, 2, N), dtype=np.float32)
            interleaved[:, :, :, 0, :] = real_part
            interleaved[:, :, :, 1, :] = imag_part
        else:
            # General case: shape becomes (x, y, z, grouped_time_steps*2, N)
            interleaved = np.empty((x, y, z, grouped_time_steps * 2, N), dtype=np.float32)
            interleaved[:, :, :, 0::2, :] = real_part
            interleaved[:, :, :, 1::2, :] = imag_part

        # Transpose to (N, grouped_time_steps*2, x, y, z)
        reshaped_data = np.transpose(interleaved, (4, 3, 0, 1, 2))
        return reshaped_data


def inverse_reshape_for_pytorch(data: np.ndarray, grouped_time_steps: int, is_mask: bool = False) -> np.ndarray:
    """
    Reverses the reshape_for_pytorch transformation using only NumPy.
    
    Parameters:
    - data: A NumPy array. If is_mask is False, expected shape is 
      (N, grouped_time_steps * 2, x, y, z); if is_mask is True, expected shape is 
      (N, 1, x, y, z).
    - grouped_time_steps: The integer 'k' used in the forward reshape (only relevant if is_mask is False).
    - is_mask: A boolean flag indicating if the input is a real-valued mask.
    
    Returns:
    - If is_mask is False: a complex NumPy array of shape (x, y, z, grouped_time_steps, N).
    - If is_mask is True: a real NumPy array of shape (x, y, z, N).
    """
    if is_mask:
        # For masks, assume data shape is (N, 1, x, y, z)
        # Inverse of the forward: (x, y, z, N) -> (N, 1, x, y, z)
        # So here we first remove the channel dimension, then invert the transpose.
        data = np.squeeze(data, axis=1)  # now shape is (N, x, y, z)
        # Inverse the transpose: forward was np.transpose(data, (3, 0, 1, 2)) to get (N, x, y, z)
        # So we do:
        data = np.transpose(data, (1, 2, 3, 0))  # now shape is (x, y, z, N)
        return data.astype(np.float32)
    else:
        # For complex data, assume data shape is (N, grouped_time_steps * 2, x, y, z)
        # The forward transformation was: (x, y, z, grouped_time_steps*2, N) -> (N, grouped_time_steps*2, x, y, z)
        # So we invert that transpose:
        data = np.transpose(data, (2, 3, 4, 1, 0))
        # Now shape is (x, y, z, grouped_time_steps*2, N)
        if grouped_time_steps == 1:
            real_part = data[..., 0, :]
            imag_part = data[..., 1, :]
        else:
            real_part = data[..., 0::2, :]
            imag_part = data[..., 1::2, :]
        complex_data = real_part + 1j * imag_part
        return complex_data



#### Functions for Fourier Transform ####

def fourier_transform(images):
    """
    Computes the 3D Fourier Transform of each 'image' along the first three dimensions.
    
    Parameters
    ----------
    images : np.ndarray
        A numpy array of shape (N1, N2, N3, ...), where (N1, N2, N3) corresponds to 
        the spatial dimensions of each 3D image. The entries may be real or complex.
        
    Returns
    -------
    f_transform_shifted : np.ndarray
        A numpy array of the same shape (N1, N2, N3, ...), containing complex values of the 
        Fourier-transformed images with zero frequencies centered.
    """
    # Perform a 2D FFT along the first two dimensions
    f_transform = np.fft.fft2(images, axes=(0, 1, 2))
    
    # Shift the zero frequency component to the center
    f_transform_shifted = np.fft.fftshift(f_transform, axes=(0, 1, 2))
    
    return f_transform_shifted

def inverse_fourier_transform(f_transform_shifted):
    """
    Computes the inverse 3D Fourier Transform of each Fourier-transformed 'image',
    undoing the operations performed by `compute_fourier_transform`.

    Parameters
    ----------
    f_transform_shifted : np.ndarray
        A numpy array of shape (N1, N2, N3, ...), where (N1, N2, N3) corresponds to 
        the Fourier-transformed spatial dimensions. The entries are complex values.
        
    Returns
    -------
    reconstructed_images : np.ndarray
        A numpy array of the same shape (N1, N2, N3, ...), containing the reconstructed 
        spatial-domain images. The result will be real if the input images to the forward 
        transform were real.
    """
    import numpy as np

    # Unshift the zero frequency component back
    f_transform_unshifted = np.fft.ifftshift(f_transform_shifted, axes=(0, 1, 2))
    
    # Perform the inverse 3D FFT along the first three dimensions
    reconstructed_images = np.fft.ifft2(f_transform_unshifted, axes=(0, 1, 2))
    
    # Return the real part (if the original data was real)
    return reconstructed_images

#### function for grouping time steps a long the T-axis, as well as undoing the grouping ####

def group_time_steps(array, grouped_time_steps):
    """
    Groups the time steps in the input array along the T dimension into overlapping windows of size k.

    Parameters
    ----------
    array : np.ndarray
        Input array of shape (x,y,z,t T, ...), where T is the time steps dimension.
    grouped_time_steps : int
        Number of time steps to group together. If grouped_time_steps=1, the function returns the array as-is.

    Returns
    -------
    grouped_array : np.ndarray
        Array with the T dimension replaced by two new dimensions (k, windows),
        where "windows" is the number of overlapping windows: T - grouped_time_steps + 1.
        Shape becomes (x,y,z,t, grouped_time_steps, windows, ...).
    """
    # Get the shape of the input array
    shape = array.shape
    T = shape[4]  # Assume T is the last dimension for simplicity

    if grouped_time_steps == 1:
        # If grouped_time_steps=1, return the array as-is
        return array

    # Compute the number of overlapping windows
    windows = T - grouped_time_steps + 1

    # Create overlapping windows along the T axis
    grouped_array = np.lib.stride_tricks.sliding_window_view(array, window_shape=(grouped_time_steps,), axis=4)
    
    # Next I make sure that the index k is next to the new T index, by default it becomes the last index with the above function
    grouped_array = np.moveaxis(grouped_array, source=-1, destination=5)
    grouped_array = np.moveaxis(grouped_array, source=4, destination=5) # I prefer (k,windows) over (windows,k)
    
    return grouped_array

def ungroup_time_steps(grouped_array, grouped_time_steps):# reverses the above function. 
    """
    Inverts the `group_time_steps` function to recover the original array.

    Parameters
    ----------
    grouped_array : np.ndarray
        An array of shape (..., grouped_time_steps, windows, ...), which is the output of
        group_time_steps(..., grouped_time_steps). The first dimensions (...) match those
        of the original array except for the time dimension.

    Returns
    -------
    original_array : np.ndarray
        The reconstructed array of shape (..., T, ...), where T = grouped_time_steps + windows - 1.
    """
    if grouped_time_steps==1:
        return grouped_array
    
    # shape = (..., grouped_time_steps, windows, ...)
    shape = grouped_array.shape

    # k is the size of each window; windows is how many windows we had
    windows = shape[5]

    # Recover the original number of time steps
    T = grouped_time_steps + windows - 1

    # Construct the output shape by replacing (k, windows) with T
    # shape[:4] => all dims up to but NOT including index 4
    # shape[6:] => any dims after index 5
    out_shape = shape[:4] + (T,) + shape[6:]

    # Initialize the output array
    original_array = np.empty(out_shape, dtype=grouped_array.dtype)

    # Loop over each window and each index within the window
    # i ranges over [0..windows-1]
    # j ranges over [0..k-1]
    #
    # i + j covers the time steps [0..T-1].
    # Because the data in the overlaps is identical to the original,
    # we can use direct assignment (last one wins) or an average.
    # They will be the same if grouped_array is unchanged.
    
    for i in range(windows):
        original_array[..., i] = grouped_array[..., 0, i]
    
    for m in range(grouped_time_steps):
        original_array[..., T-grouped_time_steps+m] = grouped_array[..., m, -1]

    return original_array

def Process_Model_Output(model, data_loader, device, original_shape, inverse_reshape, grouped_time_steps, abs_test_set = []): ## this is for a network that processes k-space and image space simulattenously (deeper)
    """
    Compute model predictions and bring them back to the original unpreprocessed format for statistics.
    Also output the ground truth in the same format.

    Parameters:
        test_loader (DataLoader): DataLoader for the test set.
        model (torch.nn.Module): The trained PyTorch model.
        device (torch.device): Device to perform computations on (e.g., 'cuda' or 'cpu').
        inverse_preprocess (function): Function to revert preprocessing on the data.
        t, T, grouped_time_steps: Parameters for inverse_preprocess function.
        abs_test_set: needed to denormalize the original normalization of data, for comparison to other models NORMALIZATION VALUES

    Returns:
        tuple: The original shape of outputs and labels after inverse preprocessing.
    """
    # Set the model to evaluation mode
    model.eval()

    # Initialize lists to store outputs and labels
    outputs_list = []

    # Disable gradient computation for efficiency
    with torch.no_grad():
        for data, _, _ in data_loader:
            # Unpack the tuple returned by the dataset
            inputs_img = data

            # Move the tensors to the appropriate device
            inputs_img = inputs_img.to(device)

            # Pass the inputs as a tuple to the model
            outputs = model((inputs_img))

            # If outputs is a tuple, extract the first element
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # Append outputs and labels to the lists
            outputs_list.append(outputs.cpu().numpy())  # Convert to numpy and move to CPU


    #     # Convert to final arrays
    outputs_array = np.concatenate(outputs_list, axis=0)

    outputs_array = inverse_reshape_for_pytorch(outputs_array, grouped_time_steps)

    denormalized_output = denormalize_data_per_image(outputs_array, abs_test_set.reshape(-1))

    ### Transform to original shape
    denormalized_output = denormalized_output.reshape(original_shape)

    ### Transform to correct output

    Model_Output = denormalized_output.transpose(inverse_reshape)

    return Model_Output


def Process_Model_Output_deeper(test_loader, model, device, trancuate_t, T, grouped_time_steps, abs_test_set = []): ## this is for a network that processes k-space and image space simulattenously (deeper)
    """
    Compute model predictions and bring them back to the original unpreprocessed format for statistics.
    Also output the ground truth in the same format.

    Parameters:
        test_loader (DataLoader): DataLoader for the test set.
        model (torch.nn.Module): The trained PyTorch model.
        device (torch.device): Device to perform computations on (e.g., 'cuda' or 'cpu').
        inverse_preprocess (function): Function to revert preprocessing on the data.
        t, T, grouped_time_steps: Parameters for inverse_preprocess function.
        abs_test_set: needed to denormalize the original normalization of data, for comparison to other models NORMALIZATION VALUES

    Returns:
        tuple: The original shape of outputs and labels after inverse preprocessing.
    """
    # Set the model to evaluation mode
    model.eval()

    # Initialize lists to store outputs and labels
    outputs_list = []
    inputs_img_list = []
    input_kspace_list = []
    labels_list = []

    # Disable gradient computation for efficiency
    with torch.no_grad():
        for data, labels, _, _ in test_loader:
            # Unpack the tuple returned by the dataset
            inputs_img, inputs_kspace = data

            # Move the tensors to the appropriate device
            inputs_img = inputs_img.to(device)
            inputs_kspace = inputs_kspace.to(device)
            labels = labels.to(device)

            # Pass the inputs as a tuple to the model
            outputs = model((inputs_img, inputs_kspace))

            # If outputs is a tuple, extract the first element
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # Append outputs and labels to the lists
            outputs_list.append(outputs.cpu().numpy())  # Convert to numpy and move to CPU
            labels_list.append(labels.cpu().numpy())   # Convert to numpy and move to CPU
            inputs_img_list.append(inputs_img.cpu().numpy())
            input_kspace_list.append(inputs_kspace.cpu().numpy())


#     # Convert to final arrays
    outputs_array = np.concatenate(outputs_list, axis=0)
    input_kspace = np.concatenate(input_kspace_list, axis=0)
    inputs_img = np.concatenate(inputs_img_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)

    outputs_array = inverse_reshape_for_pytorch(outputs_array, grouped_time_steps)
    input_kspace = inverse_reshape_for_pytorch(input_kspace, grouped_time_steps)
    inputs_img = inverse_reshape_for_pytorch(inputs_img, grouped_time_steps)
    labels_array = inverse_reshape_for_pytorch(labels_array, grouped_time_steps)
    
    denormalized_input = denormalize_data_per_image(inputs_img, abs_test_set.reshape(-1))
    denormalized_k_space = denormalize_data_per_image(input_kspace, abs_test_set.reshape(-1))
    denormalized_output = denormalize_data_per_image(outputs_array, abs_test_set.reshape(-1))
    denormalized_labels = denormalize_data_per_image(labels_array, abs_test_set.reshape(-1))

    denormalized_input = reshape_for_pytorch(inputs_img, grouped_time_steps)
    denormalized_k_space = reshape_for_pytorch(input_kspace, grouped_time_steps)
    denormalized_output = reshape_for_pytorch(denormalized_output, grouped_time_steps)
    denormalized_labels = reshape_for_pytorch(denormalized_labels, grouped_time_steps)

    denormalized_output = inverse_preprocess(denormalized_output, trancuate_t, 8, grouped_time_steps)
    denormalized_labels = inverse_preprocess(denormalized_labels, trancuate_t, 8, grouped_time_steps)

    return denormalized_output, denormalized_labels

#### normalize data optionally ####

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
                normalized_k_space[:, :, :, k, n] = input_data_k_space[:, :, :, k, n] / max_abs_value

    elif len(input_data.shape) == 4:  # 3D case (22, 22, 21, N)
        max_abs_values = np.zeros(input_data.shape[3])  # Shape (N)
        
        for n in range(input_data.shape[3]):
            max_abs_value = np.max(np.abs(input_data[:, :, :, n]))
            if max_abs_value == 0:
                raise ValueError(f"Maximum absolute value of the input data is zero for image n={n}. Normalization is not possible.")
            max_abs_values[n] = max_abs_value
            normalized_input[:, :, :, n] = input_data[:, :, :, n] / max_abs_value
            normalized_ground_truth[:, :, :, n] = ground_truth[:, :, :, n] / max_abs_value
            normalized_k_space[:, :, :, n] = input_data_k_space[:, :, :, n] / max_abs_value
    else:
        raise ValueError("Unsupported data shape. Expected 4D or 5D input.")

    return normalized_input, normalized_k_space, normalized_ground_truth, max_abs_values

def normalize_data_per_image_new(input_data, ground_truth):
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
        max_abs_values = np.zeros(input_data.shape[4])  # Shape (N,)

        for n in range(input_data.shape[4]):
            max_abs_value = np.max(np.abs(input_data[:, :, :, :, n]))  # Get max across [:,:,:,:,n]
            if max_abs_value == 0:
                max_abs_value = 1 #simply dont normalize, you are inside the mask!

            max_abs_values[n] = max_abs_value  # Store normalization factor

            # Normalize all slices at once for this n
            normalized_input[:, :, :, :, n] = input_data[:, :, :, :, n] / max_abs_value
            normalized_ground_truth[:, :, :, :, n] = ground_truth[:, :, :, :, n] / max_abs_value


    elif len(input_data.shape) == 4:  # 3D case (22, 22, 21, N)
        max_abs_values = np.zeros(input_data.shape[3])  # Shape (N)
        
        for n in range(input_data.shape[3]):
            max_abs_value = np.max(np.abs(input_data[:, :, :, n]))
            if max_abs_value == 0:
                max_abs_value = 1 #simply dont normalize, you are inside the mask!
            max_abs_values[n] = max_abs_value
            normalized_input[:, :, :, n] = input_data[:, :, :, n] / max_abs_value
            normalized_ground_truth[:, :, :, n] = ground_truth[:, :, :, n] / max_abs_value
    else:
        raise ValueError("Unsupported data shape. Expected 4D or 5D input.")

    return normalized_input, normalized_ground_truth, max_abs_values

def denormalize_data_per_image(normalized_input, max_abs_values): #invert the normalization, this is essential to compare performance across models
    """
    Denormalizes input data and ground truth using the maximum absolute values for each image.

    Parameters:
    - normalized_input (numpy.ndarray): The normalized input data array.
    - max_abs_values (numpy.ndarray): The maximum absolute values used for normalization, shape (k, N) or (N).

    Returns:
    - denormalized_input (numpy.ndarray): Denormalized input data.
    """
    # Initialize denormalized arrays with the same shape as the normalized data
    denormalized_input = np.zeros_like(normalized_input)

    # Denormalize each image independently
    if len(normalized_input.shape) == 5:  # 4D case (22, 22, 21, k, N)
        for n in range(normalized_input.shape[4]):
            max_abs_value = max_abs_values[n]
            denormalized_input[:, :, :, :, n] = normalized_input[:, :, :, :, n] * max_abs_value

    elif len(normalized_input.shape) == 4:  # 3D case (22, 22, 21, N)
        for n in range(normalized_input.shape[3]):
            max_abs_value = max_abs_values[n]
            denormalized_input[:, :, :, n] = normalized_input[:, :, :, n] * max_abs_value

    else:
        raise ValueError("Unsupported data shape. Expected 4D or 5D input.")

    return denormalized_input


def low_rank(data, rank):
    """
    Computes a low-rank decomposition of a tensor with shape (22, 22, 21, 96, 8)
    using truncated SVD.

    Args:
        data (np.ndarray): Numpy array of shape (x, y, z, t, T).
        rank (int): The number of singular values to keep (final rank).

    Returns:
        np.ndarray: The reconstructed tensor with rank 'rank'.
    """

    # Unpack dimensions
    x, y, z, t, T = data.shape
    
    # Reshape the 5D tensor into a 2D matrix of shape (x*y*z, t*T)
    # Use 'F' (Fortran) order to match MATLAB's column-major ordering
    reshaped_matrix = data.reshape((x * y * z * T, t), order='F')
    
    # Perform economy-size SVD (similar to MATLAB's "svd(..., 'econ')")
    U, singular_values, Vh = np.linalg.svd(reshaped_matrix, full_matrices=False)
    
    # Truncate the singular values to the desired rank
    k = min(rank, len(singular_values))  # safeguard: rank cannot exceed # of singular values
    singular_values_truncated = np.zeros_like(singular_values)
    singular_values_truncated[:k] = singular_values[:k]
    
    # Form the diagonal matrix of truncated singular values
    S_truncated = np.diag(singular_values_truncated)
    
    # Reconstruct the matrix using the truncated SVD components
    reconstructed_matrix = U @ S_truncated @ Vh
    
    # Reshape back to the original 5D shape, again using 'F' order
    reconstructed_tensor = reconstructed_matrix.reshape((x, y, z, t, T), order='F')
    
    return reconstructed_tensor
    
def normalize_data(input_data, ground_truth):
    """
    Normalizes input data and ground truth on a per-image basis to the range [-1, 1]
    using the maximum absolute value of each image.

    Parameters:
    - input_data (numpy.ndarray): The input data array of shape (22, 22, 21, t, T, D).
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
    max_abs_values = np.zeros((input_data.shape[3], input_data.shape[4], input_data.shape[5]))  # Shape (k, N)
    for t in range(input_data.shape[3]):
        for T in range(input_data.shape[4]):
            for D in range(input_data.shape[5]):
                max_abs_value = np.max(np.abs(input_data[:, :, :, t, T, D]))
                max_abs_values[t, T, D] = max_abs_value
                normalized_input[:, :, :, t, T, D] = input_data[:, :, :, t, T, D] / max_abs_value
                normalized_ground_truth[:, :, :, t, T, D] = ground_truth[:, :, :, t, T, D] / max_abs_value


    return normalized_input, normalized_ground_truth, max_abs_values

def convert_complex_to_real(array):
    """
    Takes a complex numpy array, separates real and imaginary part
    and stacks them along the last axis.
    
    Args:
        numpy array with complex numbers.
        
    Returns:
        Stacked array with real and imaginary part.
    """
    array_real = array.real
    array_img = array.imag
    stacked_array = np.stack((array_real, array_img), axis=-1)
    
    return stacked_array

def apply_masks(data, masks):
    """
    Applies patient-specific masks to the data.
    
    Parameters:
      data: numpy array of shape (22, 22, 21, 96, 8, 6)
            The first three dimensions are spatial, followed by time dimensions, and
            the last dimension indexes the patient.
      masks: numpy array of shape (22, 22, 21, 6)
             Each mask corresponds to a patient and has the same spatial dimensions.
             
    Returns:
      masked_data: numpy array of the same shape as data, with each patient's
                   data masked by the corresponding mask.
    """
    # Expand the masks to have singleton dimensions for the time axes.
    # New shape: (22, 22, 21, 1, 1, 6)
    masks_expanded = masks[:, :, :, np.newaxis, np.newaxis, :]
    
    # Multiply the data by the expanded masks.
    masked_data = data * masks_expanded
    
    return masked_data

def reshape_for_pytorch_2D(data):
    """
    Reshapes complex-valued MRSI data into a format suitable for PyTorch CNN training.

    Parameters:
    - data: numpy array of shape (x, y, grouped_time_steps, N), where:
        * (x, y) are spatial dimensions (image size),
        * grouped_time_steps is the number of grouped time steps,
        * N is the number of samples (data points).

    Returns:
    - reshaped_data: PyTorch tensor of shape (N, grouped_time_steps * 2, x, y, z).
    """
    #x, y, z = data.shape[0], data.shape[1], data.shape[2]
    # Separate real and imaginary parts
    N = data.shape[-1]
    real_part = np.real(data)
    imag_part = np.imag(data)
    
    x = data.shape[0]
    y = data.shape[1]

    # Interleave real and imaginary parts along the feature axis

        # Handle the special case for k = 1
    interleaved = np.empty((x, y, 2, N), dtype=np.float32)
    interleaved[:,:, 0, :] = real_part  # Real part
    interleaved[:,:, 1, :] = imag_part  # Imaginary part

    # Transpose to (N, k*2, 22, 22, 21) for PyTorch
    reshaped_data = np.transpose(interleaved, (3, 2, 1, 0))  # Shape: (N, k*2, 22, 22, 21)

    return reshaped_data
    
def reshape_for_pytorch_2D_TEST(data, truncate_t):
    """
    Reshapes complex-valued MRSI data into a format suitable for PyTorch CNN training.

    Parameters:
    - data: numpy array of shape (x, y, z, grouped_time_steps, N), where:
        * (x, y) are spatial dimensions (image size),
        * 21 is the number of slices along the z-axis, or T alternatively,
        * grouped_time_steps is the number of grouped time steps,
        * N is the number of samples (data points).

    Returns:
    - reshaped_data: PyTorch tensor of shape (N, grouped_time_steps * 2, x, y, z).
    """
    #x, y, z = data.shape[0], data.shape[1], data.shape[2]
    # Separate real and imaginary parts
    N = data.shape[-1]
    real_part = np.real(data)
    imag_part = np.imag(data)

    # Interleave real and imaginary parts along the feature axis

        # Handle the special case for k = 1
    interleaved = np.empty((data.shape[0], data.shape[1], 2, N), dtype=np.float32)
    interleaved[:,:, 0, :] = real_part  # Real part
    interleaved[:,:, 1, :] = imag_part  # Imaginary part

    # Transpose to (N, k*2, 22, 22, 21) for PyTorch
    reshaped_data = np.transpose(interleaved, (3, 2, 1, 0))  # Shape: (N, k*2, 22, 22, 21)

    return reshaped_data
    
import numpy as np

def compute_relative_coords_per_axis_clamped(MASKS):
    """
    Given a head mask of shape (X, Y, Z, P), computes two arrays:
      - rel_x: for each z-slice and row y, we find x_min, x_max where mask>0,
               then for each x:
                 if x < x_min or x > x_max => -1,
                 else => (x - x_min)/(x_max - x_min).
      - rel_y: for each z-slice and column x, similarly for y.
    This ensures out-of-bounds coordinates are -1, and valid ones are in [0,1].
    
    Args:
        MASKS: np.ndarray of shape (X, Y, Z, P).
    
    Returns:
        rel_x, rel_y: 
            two np.float32 arrays, each shape (X, Y, Z, P),
            with values in [-1, 0..1].
            -1 => outside the bounding box for that row/column/slice,
            [0..1] => normalized coordinate if x_min<x_max or y_min<y_max.
    """
    X, Y, Z, P = MASKS.shape
    
    rel_x = np.zeros((X, Y, Z, P), dtype=np.float32)
    rel_y = np.zeros((X, Y, Z, P), dtype=np.float32)
    
    # ---------- Compute rel_x ----------
    for p_idx in range(P):
        for z_idx in range(Z):
            for y_idx in range(Y):
                slice_x = MASKS[:, y_idx, z_idx, p_idx]  # shape: (X,)
                indices = np.argwhere(slice_x > 0).flatten()
                if indices.size == 0:
                    # Entire row invalid => -1
                    rel_x[:, y_idx, z_idx, p_idx] = -1
                else:
                    x_min = np.min(indices)
                    x_max = np.max(indices)
                    if x_max > x_min:
                        for i_idx in range(X):
                            if i_idx < x_min or i_idx > x_max:
                                rel_x[i_idx, y_idx, z_idx, p_idx] = -1
                            else:
                                # Normalize to [0,1]
                                rel_x[i_idx, y_idx, z_idx, p_idx] = \
                                    (i_idx - x_min) / float(x_max - x_min)
                    else:
                        # row has only one valid x => set it 0, everything else -1
                        for i_idx in range(X):
                            if i_idx == x_min:
                                rel_x[i_idx, y_idx, z_idx, p_idx] = 0
                            else:
                                rel_x[i_idx, y_idx, z_idx, p_idx] = -1
    
    # ---------- Compute rel_y ----------
    for p_idx in range(P):
        for z_idx in range(Z):
            for x_idx in range(X):
                slice_y = MASKS[x_idx, :, z_idx, p_idx]  # shape: (Y,)
                indices = np.argwhere(slice_y > 0).flatten()
                if indices.size == 0:
                    rel_y[x_idx, :, z_idx, p_idx] = -1
                else:
                    y_min = np.min(indices)
                    y_max = np.max(indices)
                    if y_max > y_min:
                        for j_idx in range(Y):
                            if j_idx < y_min or j_idx > y_max:
                                rel_y[x_idx, j_idx, z_idx, p_idx] = -1
                            else:
                                rel_y[x_idx, j_idx, z_idx, p_idx] = \
                                    (j_idx - y_min) / float(y_max - y_min)
                    else:
                        # column has only one valid y => 0 at that position, -1 elsewhere
                        for j_idx in range(Y):
                            if j_idx == y_min:
                                rel_y[x_idx, j_idx, z_idx, p_idx] = 0
                            else:
                                rel_y[x_idx, j_idx, z_idx, p_idx] = -1
    
    return rel_x, rel_y

import numpy as np

def get_reshape_vectors(domain: str):
    """
    Returns the reshape and inverse reshape vectors for the given domain.
    
    Domains:
      - "kzfT": reshape_vector = (2, 3, 4, 0, 1, 5)
                inverse_reshape = (3, 4, 0, 1, 2, 5)
      - "zfT":  reshape_vector = (2, 3, 4, 0, 1, 5)
                inverse_reshape = (3, 4, 0, 1, 2, 5)
      - "ztT":  reshape_vector = (2, 3, 4, 0, 1, 5)
                inverse_reshape = (3, 4, 0, 1, 2, 5)
      - "xyz":  reshape_vector = (0, 1, 2, 3, 4, 5)
                inverse_reshape = (0, 1, 2, 3, 4, 5)
      - "xzT":  reshape_vector = (0, 2, 4, 3, 1, 5)
                inverse_reshape = (0, 4, 1, 3, 2, 5)

    Parameters:
        domain (str): One of "kzfT", "zfT", "ztT", "xyz", or "xzT".

    Returns:
        tuple: (reshape_vector, inverse_reshape)
    
    Raises:
        ValueError: If an invalid domain is provided.
    """
    if domain == "kzfT":
        reshape_vector = (2, 3, 4, 0, 1, 5)
        inverse_reshape = (3, 4, 0, 1, 2, 5)
    elif domain == "zfT":
        reshape_vector = (2, 3, 4, 0, 1, 5)
        inverse_reshape = (3, 4, 0, 1, 2, 5)
    elif domain == "ztT":
        reshape_vector = (2, 3, 4, 0, 1, 5)
        inverse_reshape = (3, 4, 0, 1, 2, 5)
    elif domain == "xyz":
        reshape_vector = (0, 1, 2, 3, 4, 5)
        inverse_reshape = (0, 1, 2, 3, 4, 5)
    elif domain == "xzT":
        reshape_vector = (0, 2, 4, 3, 1, 5)
        inverse_reshape = (0, 4, 1, 3, 2, 5)
    elif domain == "xyf":
        reshape_vector = (0, 1, 3, 2, 4, 5)
        inverse_reshape = (0, 1, 3, 2, 4, 5)
    else:
        raise ValueError("No valid Domain chosen!")
    
    return reshape_vector, inverse_reshape


def apply_domain_transforms(domain: str, undersampled_data: np.ndarray, ground_truth: np.ndarray):
    """
    Applies FFT transforms to the data based on the domain.
    
    Domain-specific operations:
      - "kzfT": Apply FFT (with fftshift) along axis -3, then along axis 3.
      - "zfT":  Apply FFT (with fftshift) along axis -3 only.
      - "ztT":  No FFT transformation.
      - "xyz":  No FFT transformation.
      - "xzT":  No FFT transformation.

    Parameters:
        domain (str): One of "kzfT", "zfT", "ztT", "xyz", or "xzT".
        undersampled_data (np.ndarray): The undersampled data array.
        ground_truth (np.ndarray): The ground truth data array.

    Returns:
        tuple: (transformed_undersampled_data, transformed_ground_truth)
    
    Raises:
        ValueError: If an invalid domain is provided.
    """
    if domain == "kzfT":
        # Perform spectral transformation along axis -3
        undersampled_data = np.fft.fftshift(np.fft.fft(undersampled_data, axis=-3), axes=-3)
        ground_truth = np.fft.fftshift(np.fft.fft(ground_truth, axis=-3), axes=-3)
        # Additional transformation to k-space along axis 3
        undersampled_data = np.fft.fftshift(np.fft.fft(undersampled_data, axis=3), axes=3)
        ground_truth = np.fft.fftshift(np.fft.fft(ground_truth, axis=3), axes=3)
    elif domain == "zfT" or domain == "xyf":
        # Perform spectral transformation along axis -3 only
        undersampled_data = np.fft.fftshift(np.fft.fft(undersampled_data, axis=-3), axes=-3)
        ground_truth = np.fft.fftshift(np.fft.fft(ground_truth, axis=-3), axes=-3)
    elif domain == "ztT":
        # No FFT transformation is performed for "ztT"
        pass
    elif domain == "xyz":
        # No FFT transformation is performed for "xyz"
        pass
    elif domain == "xzT":
        # No FFT transformation is performed for "xzT"
        pass
    else:
        raise ValueError("No valid Domain chosen!")
    
    return undersampled_data, ground_truth


# Example usage:
if __name__ == "__main__":
    # Define your domain, undersampled data, and ground truth arrays.
    domain = "kzfT"  # Change to any valid domain: "kzfT", "zfT", "ztT", "xyz", or "xzT"
    
    # Example arrays (replace these with your actual data)
    undersampled_data = np.random.rand(10, 10, 10, 10, 10, 10)
    ground_truth = np.random.rand(10, 10, 10, 10, 10, 10)

    # Get the reshape vectors for the specified domain.
    reshape_vector, inverse_reshape = get_reshape_vectors(domain)
    print("Reshape Vector:", reshape_vector)
    print("Inverse Reshape Vector:", inverse_reshape)

    # Apply the domain-specific FFT transforms.
    transformed_us_data, transformed_gt = apply_domain_transforms(domain, undersampled_data, ground_truth)


