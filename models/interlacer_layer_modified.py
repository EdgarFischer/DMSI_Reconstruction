#### I modify the interlacer layer to match exactly the description in the paper for my case, I want to compare this to my Naive_CNN_3D_Residual

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class TensorDataset_interlacer(Dataset):
    def __init__(self, k_space, image_reconstructed, ground_truth, masks, norm_values):
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
        self.masks = torch.tensor(masks, dtype=torch.float32)
        
        # Convert norm_values to a tensor if it's a list or numpy array.
        if isinstance(norm_values, (list, np.ndarray)):
            self.norm_values = torch.tensor(norm_values, dtype=torch.float32)
        else:
            self.norm_values = norm_values

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
        mask = self.masks[idx]
        norm_value = self.norm_values[idx]

        inputs = (inputs_img, inputs_kspace)
        ground_truth = self.ground_truth[idx]

        return inputs, ground_truth, mask, norm_value


def piecewise_relu(x):
    """Custom nonlinearity for freq-space convolutions."""
    return x + F.relu(1 / 2 * (x - 1)) + F.relu(1 / 2 * (-1 - x))


def get_nonlinear_layer(nonlinearity):
    """Selects and returns an appropriate nonlinearity."""
    if(nonlinearity == 'relu'):
        return torch.nn.ReLU()
    elif(nonlinearity == '3-piece'):
        return piecewise_relu

class BatchNormConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, use_norm):
        super().__init__()
        self.in_channels = in_channels
        
        self.in_channels_conv = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.use_norm = use_norm

        if use_norm=="BatchNorm":
            self.bn = nn.BatchNorm3d(
                num_features = self.in_channels)
        elif use_norm=="InstanceNorm":
            self.bn = nn.InstanceNorm3d(
                num_features = self.in_channels)
        elif use_norm=="None":
            self.bn = None
        else:
            print("Pick an available Normalizaiton Method")
            sys.exit()
        
        self.conv = nn.Conv3d(
            in_channels = self.in_channels_conv,
            out_channels = self.out_channels,
            kernel_size=self.kernel_size,
            padding="same")
    
    def forward(self, x):
        """Core layer function to combine BN/convolution.
        Args:
          x: Input tensor
        Returns:
          conv(float): Output of BN (on axis 0) followed by convolution
        """
        
        if not self.use_norm=="None":
            x = self.bn(x)
        x = self.conv(x)

        return x
    """
    def compute_output_shape(self, input_shape):
        return (input_shape[:3] + [self.features])
    """

class Mix(nn.Module):
    """Custom layer to learn a combination of two inputs."""
    def __init__(self):
        super().__init__()
        self._mix = nn.Parameter(torch.rand((1,)), requires_grad=True)
        
    def forward(self, x):
        """Core layer function to combine inputs.
        Args:
          x: Tuple (A,B), where A and B are numpy arrays of equal shape
        Returns:
          sig_mix*A + (1-sig_mix)B, where six_mix = sigmoid(mix) and mix is a learned combination parameter
        """
        A, B = x
        sig_mix = torch.sigmoid(self._mix)
        return sig_mix * A + (1 - sig_mix) * B
    
    """
    def compute_output_shape(self, input_shape):
        return input_shape[0]
    """



class Interlacer_Modified(nn.Module):
    """Custom layer to learn features in both image and frequency space."""
    def __init__(self, features_img, features_kspace, kernel_size, 
                        use_norm, num_convs=3, shift=False):
        super().__init__()
        self.features_img = features_img
        self.features_kspace = features_kspace
        self.kernel_size = kernel_size
        self.num_convs = num_convs
        self.use_norm = use_norm
        
        self.img_mix = Mix()
        self.freq_mix = Mix()
        ImgModuleList = []
        FreqModuleList = []
        for i in range(self.num_convs): #### last layer is treated separately
            if i == 0:
                ImgModuleList.append(BatchNormConv(2, self.features_img, self.kernel_size, self.use_norm)) # the *2 likely reflects the operation img_feat = torch.cat((img_feat, inputs_img), dim=1) in the forwar func.
                FreqModuleList.append(BatchNormConv(2, self.features_kspace, self.kernel_size, self.use_norm)) # the *2 likely reflects k_feat = torch.cat((k_feat, inputs_kspace), dim=1) .... see above
            elif i == 1:
                ImgModuleList.append(BatchNormConv(self.features_img, self.features_img, self.kernel_size, self.use_norm))
                FreqModuleList.append(BatchNormConv(self.features_kspace, self.features_kspace, self.kernel_size, self.use_norm))
            elif i == 2:
                ImgModuleList.append(BatchNormConv(self.features_img, 2, self.kernel_size, self.use_norm))
                FreqModuleList.append(BatchNormConv(self.features_kspace, 2, self.kernel_size, self.use_norm))
        
        self.img_bnconvs = nn.ModuleList(ImgModuleList)
        self.freq_bnconvs = nn.ModuleList(FreqModuleList)
        
#         # Add final layers to reduce channels to 2
#         self.final_img_conv = BatchNormConv(self.features_img, 2, self.kernel_size, self.use_norm)  # Reduce img channels to 2
#         self.final_kspace_conv = BatchNormConv(self.features_kspace, 2, self.kernel_size, self.use_norm)  # Reduce kspace channels to 2
        
    def forward(self, x):
        """Core layer function to learn image and frequency features.
        Args:
          x: Tuple (A,B), where A contains image-space features and B contains frequency-space features
        Returns:
          img_conv(float): nonlinear(conv(BN(beta*img_in+IFFT(freq_in))))
          freq_conv(float): nonlinear(conv(BN(alpha*freq_in+FFT(img_in))))
        """
        
        img_in, freq_in = x # note that inputs_img, inputs_kspace are the original input to the network, not just the input to the layer

        
        batchsz = img_in.shape[0]
        # Initialize lists to store the results
        img_in_as_freq_list = []
        freq_in_as_img_list = []

        # Loop through the batch and compute FFT/IFFT
        for i in range(batchsz):
            # Combine real and imaginary parts into complex tensor for FFT
            img_in_complex = img_in[i, 0] + 1j * img_in[i, 1]  # Assumes channels [real, imag]
            freq_in_complex = freq_in[i, 0] + 1j * freq_in[i, 1]

            # Perform FFT/IFFT
            img_in_as_freq = torch.fft.fftshift(torch.fft.fftn(img_in_complex, dim=(0, 1, 2)), dim=(0, 1, 2))
            freq_in_as_img = torch.fft.ifftn(torch.fft.ifftshift(freq_in_complex, dim=(0, 1, 2)), dim=(0, 1, 2))

            # Convert back to real-valued tensors (real + imaginary channels)
            img_in_as_freq = torch.stack([img_in_as_freq.real, img_in_as_freq.imag], dim=0)
            freq_in_as_img = torch.stack([freq_in_as_img.real, freq_in_as_img.imag], dim=0)

            img_in_as_freq_list.append(img_in_as_freq)
            freq_in_as_img_list.append(freq_in_as_img)
                

        img_feat = self.img_mix([img_in, freq_in_as_img]) # this is the mixed representation of the transformed freq. domain and original image domain
        k_feat = self.freq_mix([freq_in, img_in_as_freq]) # this is the mixed representation of the transformed img. domain and original freq. domain

        #img_feat = torch.cat((img_feat, inputs_img), dim=1) # This is not described in the paper
        #k_feat = torch.cat((k_feat, inputs_kspace), dim=1)    
        
        for i in range(self.num_convs):
            img_conv = self.img_bnconvs[i](img_feat)
            img_feat = get_nonlinear_layer('relu')(img_conv)

            k_conv = self.freq_bnconvs[i](k_feat)
            k_feat = get_nonlinear_layer('3-piece')(k_conv)
        
        # Apply final layers to reduce channels
        #final_img_conv
        
        #img_feat = self.final_img_conv(img_feat)+img_in
        #k_feat = self.final_kspace_conv(k_feat)+freq_in
        
        return (img_feat, k_feat)



class CustomLoss(torch.nn.Module):
    """Loss function that computes MSE in image space."""
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs, targets):
        """
        Args:
          outputs: Predicted images from the model.
          targets: Ground truth images.
        Returns:
          MSE loss between predicted and ground truth images.
        """
        return self.mse_loss(outputs, targets)
    
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
    
    # Return the average loss for the epoch
    return total_loss / num_samples



def validate_model(model, loss_fn, data_loader, device='cpu'):
    """
    Validates the model on the given DataLoader, applying a mask to 
    focus the loss on valid regions in image space.
    
    Args:
        model (nn.Module): The model to validate.
        loss_fn (callable): The loss function to compute the error (e.g., nn.MSELoss()).
        data_loader (DataLoader): DataLoader for the validation dataset.
        device (str): Device to run the validation on ('cpu' or 'cuda').

    Returns:
        float: Average validation loss for the dataset.
    """
    model.eval()  # Set model to evaluation mode
    
    total_loss = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for inputD, dat_out, mask, norm_value in data_loader:
            # Move inputs and targets to the device
            img_in, freq_in = inputD
            img_in   = img_in.to(device)
            freq_in  = freq_in.to(device)
            dat_out  = dat_out.to(device)
            mask     = mask.to(device)
            norm_value = norm_value.to(device)

            # Prepare the input tuple for the model
            inputs = (img_in, freq_in)
            
            # Forward pass
            predictions, _ = model(inputs)
            # Reshape norm_value to (N, 1, 1, 1, 1) so it broadcasts properly
            norm_value = norm_value.view(norm_value.size(0), 1, 1, 1, 1)
            
            # Apply the mask to both predictions and ground truth
            masked_predictions = predictions * mask * norm_value
            masked_targets     = dat_out * mask * norm_value
            
            # Compute loss only on masked data in image space
            loss_curr = loss_fn(masked_predictions, masked_targets)
            
            batch_size = dat_out.size(0)
            total_loss += loss_curr.item() * batch_size
            num_samples += batch_size
    
    return total_loss / num_samples






