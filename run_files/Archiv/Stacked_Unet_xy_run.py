import os
import sys

sys.path.append('../scripts')
sys.path.append('../models')

os.environ["CUDA_VISIBLE_DEVICES"]= '1' #, this way I would choose GPU 3 to do the work

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom # for compressing images / only for testing purposes to speed up NN training
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch.nn as nn
from data_preparation import *
from data_undersampling import *
from Stacked_Unet_xy import *
from output_statistics import *

from data_augmentation_xyztT import *

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio 

grouped_time_steps = 1 # Set how many subsequent time steps you want to give to the network at once. Values allowed: 1, 2, 4, 8 (because it has to divide 8)

######## SET PARAMETERS ########
#### Undersampling Strategy:#####
#Real_Undersampling = True ## If True is chosen, the undersampling data that is directly obtained from undersampling on the raw ring structure is taken with non uniform FT.
Undersampling = "Possoin_Real" # Options: Regular or Possoin, Regular_Real, Possoin_Real
Sampling_Mask = "Complementary_Masks" #Options: Single_Combination or One_Mask or Complementary_Masks
AF = 5 #  acceleration factor
DOMAIN = "zfT"# Input axes that goes into the network (as well as output), 
            # valid options: kzfT (this means k_z in kspace); zfT; ztT; xyz; xzT

#### Model Input and Output ####
GT_Data = "LowRank" # Options: FullRank LowRank for GROUNDTRUTH! 
Low_Rank_Input = True ## apply low rank to the input as well if True

batch_size=125
num_epochs = 1000
print_every = 1
trancuate_t = 96
Layers = 5

#### Parameter setting END ####

#### Define ground truth path####
ground_truth_path = "../data/Ground_Truth/Full_Rank/Full_Rank_All.npy"
    
#### Assemble saving path ####

#### Define Model path
if GT_Data == "FullRank":
    model_save_dir = f"../saved_models/Stacked_Unet_xy/{DOMAIN}/Full2Full/{Undersampling}/AF_{AF}/Truncate_t_{trancuate_t}/{Sampling_Mask}/Layers_{Layers}"
elif GT_Data == "LowRank":
    model_save_dir = f"../saved_models/Stacked_Unet_xy/{DOMAIN}/Low2Low/{Undersampling}/AF_{AF}/Truncate_t_{trancuate_t}/{Sampling_Mask}/Layers_{Layers}"
    
#### Define log directory path
if GT_Data == "FullRank":
    log_dir = f"../log_files/Stacked_Unet_xy/{DOMAIN}/Full2Full/{Undersampling}/AF_{AF}/Truncate_t_{trancuate_t}/{Sampling_Mask}/Layers_{Layers}"
elif GT_Data == "LowRank":
    log_dir = f"../log_files/Stacked_Unet_xy/{DOMAIN}/Low2Low/{Undersampling}/AF_{AF}/Truncate_t_{trancuate_t}/{Sampling_Mask}/Layers_{Layers}"

#### Define Input Data path
undersampled_data_path = "../data/Undersampled_Data/"+Undersampling+f'/AF_{AF}/'+Sampling_Mask+'/data.npy'

#### load data!
Ground_Truth = np.load(ground_truth_path)
Undersampled_Data = np.load(undersampled_data_path)

if Low_Rank_Input:
    for i in range(0,6):
        Undersampled_Data[...,i] = low_rank(Undersampled_Data[...,i], 8)
        if GT_Data == "LowRank":
            Ground_Truth[...,i] = low_rank(Ground_Truth[...,i], 8)
    
###normalize entire 5D volumes volumes 
for i in range(0,6):
    Ground_Truth[...,i] = Ground_Truth[...,i]/np.max(np.abs(Ground_Truth[...,i]))
    Undersampled_Data[...,i] = Undersampled_Data[...,i]/np.max(np.abs(Undersampled_Data[...,i]))

MASKS = np.load("../data/masks.npy")
rel_x, rel_y = compute_relative_coords_per_axis_clamped(MASKS)

####
mask_expanded = MASKS[:, :, :, None, None, :]  # Now shape is (22,22,21,1,1,6)
# Use broadcasting to "repeat" the mask along these new axes:
mask_extended = np.broadcast_to(mask_expanded, (22, 22, 21, 96, 8, 6))
#mask_extended = mask_extended + 1J*mask_extended

rel_x_expanded, rel_y_expanded = rel_x[:,:,:,None, None, :], rel_y[:,:,:,None, None, :]
rel_x_expanded, rel_y_expanded = np.broadcast_to(rel_x_expanded, (22, 22, 21, 96, 8, 6)), np.broadcast_to(rel_y_expanded, (22, 22, 21, 96, 8, 6))

#### trancuate spectral dimension with trancuate_t
Ground_Truth, Undersampled_Data, mask_extended = Ground_Truth[:,:,:,:trancuate_t,:,:], Undersampled_Data[:,:,:,:trancuate_t,:,:], mask_extended[:,:,:,:trancuate_t,:,:] 
    
### Pad spatial dimensions from 22,22,21 to 24,24,24 which is necessary for U-Net (needs to be divisible by 8 due to downsampling)
pad_width = ((1, 1), (1, 1), (1, 2),(0, 0), (0, 0), (0, 0))    

Ground_Truth = np.pad(Ground_Truth, pad_width, mode='constant', constant_values=0)
Undersampled_Data = np.pad(Undersampled_Data, pad_width, mode='constant', constant_values=0)
mask_extended = np.pad(mask_extended, pad_width, mode='constant', constant_values=0)   
rel_x_expanded = np.pad(rel_x_expanded, pad_width, mode='constant', constant_values=-1)
rel_y_expanded = np.pad(rel_y_expanded, pad_width, mode='constant', constant_values=-1)

if DOMAIN == "kzfT":     
    ### spectral transformaton
    Undersampled_Data = np.fft.fftshift(np.fft.fft(Undersampled_Data, axis=-3), axes=-3)
    Ground_Truth = np.fft.fftshift(np.fft.fft(Ground_Truth, axis=-3), axes=-3)    
    #### trafo to k-space
    Undersampled_Data = np.fft.fftshift(np.fft.fft(Undersampled_Data, axis=3), axes=3)
    Ground_Truth = np.fft.fftshift(np.fft.fft(Ground_Truth, axis=3), axes=3) 
    #### reshap vectors
    reshape_vector = (2, 3, 4, 0, 1, 5)
    inverse_reshape = (3, 4, 0, 1, 2, 5)
    
elif DOMAIN == "zfT":
    ### spectral transformaton
    Undersampled_Data = np.fft.fftshift(np.fft.fft(Undersampled_Data, axis=-3), axes=-3)
    Ground_Truth = np.fft.fftshift(np.fft.fft(Ground_Truth, axis=-3), axes=-3)   
    #### reshap vectors
    reshape_vector = (2, 3, 4, 0, 1, 5)
    inverse_reshape = (3, 4, 0, 1, 2, 5)
    
elif DOMAIN == "ztT":   
    #### reshap vectors
    reshape_vector = (2, 3, 4, 0, 1, 5)
    inverse_reshape = (3, 4, 0, 1, 2, 5)
    
elif DOMAIN == "xyz":   
    #### reshap vectors
    reshape_vector = (0, 1, 2, 3, 4, 5)
    inverse_reshape = (0, 1, 2, 3, 4, 5)
    
elif DOMAIN == "xzT":   
    #### reshap vectors
    reshape_vector = (0, 2, 4, 3, 1, 5)
    inverse_reshape = (0, 4, 1, 3, 2, 5)
    
else: 
    raise Exception("No valid Domain chosen!")
    
#### Reshaping !
Undersampled_Data, Ground_Truth, mask_extended, rel_x_expanded, rel_y_expanded = Undersampled_Data.transpose(reshape_vector), Ground_Truth.transpose(reshape_vector), mask_extended.transpose(reshape_vector), rel_x_expanded.transpose(reshape_vector), rel_y_expanded.transpose(reshape_vector)

original_shape = Ground_Truth.copy().shape

##### test ende ####
ground_truth_val, ground_truth_train, ground_truth_test = Ground_Truth[..., 5], Ground_Truth[...,0:5].transpose(inverse_reshape), Ground_Truth[...,5]  # Method: Leave last MRSI measurement as test set
Val_Mask, Train_Mask, Test_Mask = mask_extended[...,5], mask_extended[...,0:5].transpose(inverse_reshape), mask_extended[...,5]

Val_rel_x, Train_rel_x, Test_rel_x = rel_x_expanded[...,5], rel_x_expanded[...,0:5].transpose(inverse_reshape), rel_x_expanded[...,5]
Val_rel_y, Train_rel_y, Test_rel_y = rel_y_expanded[...,5], rel_y_expanded[...,0:5].transpose(inverse_reshape), rel_y_expanded[...,5]

#### Assign undersampled network input ####
NN_input_val, NN_input_train, NN_input_test = Undersampled_Data[...,5], Undersampled_Data[...,0:5].transpose(inverse_reshape), Undersampled_Data[...,5]

#### Collapse ununsed dimensions ####
ground_truth_val, ground_truth_test = ground_truth_val.reshape(Ground_Truth.shape[0], Ground_Truth.shape[1], Ground_Truth.shape[2], -1), ground_truth_test.reshape(Ground_Truth.shape[0], Ground_Truth.shape[1], Ground_Truth.shape[2], -1)

NN_input_val, NN_input_test = NN_input_val.reshape(Ground_Truth.shape[0], Ground_Truth.shape[1], Ground_Truth.shape[2], -1), NN_input_test.reshape(Ground_Truth.shape[0], Ground_Truth.shape[1], Ground_Truth.shape[2], -1)

Mask_val, Mask_test = Val_Mask.reshape(Ground_Truth.shape[0], Ground_Truth.shape[1], Ground_Truth.shape[2], -1), Test_Mask.reshape(Ground_Truth.shape[0], Ground_Truth.shape[1], Ground_Truth.shape[2], -1)

Val_rel_x, Test_rel_x = Val_rel_x.reshape(Ground_Truth.shape[0], Ground_Truth.shape[1], Ground_Truth.shape[2], -1), Test_rel_x.reshape(Ground_Truth.shape[0], Ground_Truth.shape[1], Ground_Truth.shape[2], -1)

Val_rel_y, Test_rel_y = Val_rel_y.reshape(Ground_Truth.shape[0], Ground_Truth.shape[1], Ground_Truth.shape[2], -1), Test_rel_y.reshape(Ground_Truth.shape[0], Ground_Truth.shape[1], Ground_Truth.shape[2], -1)



#### Normalize data #####
# normalized_input_val, normalized_ground_truth_val, norm_values_val = normalize_data_per_image_new(NN_input_val, ground_truth_val)
#normalized_input_train, normalized_ground_truth_train, norm_values_train = normalize_data_per_image_new(NN_input_train, ground_truth_train)
# normalized_input_test, normalized_ground_truth_test, norm_values_test = normalize_data_per_image_new(NN_input_test, ground_truth_test)

#### reshape for pytorch ####
val_data, val_labels = reshape_for_pytorch(NN_input_val, grouped_time_steps), reshape_for_pytorch(ground_truth_val, grouped_time_steps)
#train_data, train_labels  = reshape_for_pytorch(normalized_input_train, grouped_time_steps), reshape_for_pytorch(normalized_ground_truth_train, grouped_time_steps)
test_data, test_labels = reshape_for_pytorch(NN_input_test, grouped_time_steps), reshape_for_pytorch(ground_truth_test, grouped_time_steps)

val_mask,  test_mask = reshape_for_pytorch(Mask_val, grouped_time_steps), reshape_for_pytorch(Mask_test, grouped_time_steps)

Val_rel_x, Test_rel_x = reshape_for_pytorch(Val_rel_x, grouped_time_steps, is_mask=True), reshape_for_pytorch(Test_rel_x, grouped_time_steps, is_mask=True)

Val_rel_y, Test_rel_y = reshape_for_pytorch(Val_rel_y, grouped_time_steps, is_mask=True), reshape_for_pytorch(Test_rel_y, grouped_time_steps, is_mask=True)



#### prepare data for NN #####


# # Import the augmentation class !!!! CAREFUL TENSORDATA SET IS NOT COMPATIBLE WITH AUGMENTATIONS ANYMORE
# augment = RandomAugment3D(
#     rotation_range=0.3,   # Rotate by up to ±17° in xy and xz planes
#     shift_pixels=1,       # Shift up to 2 pixels in any direction
#     scale_range=0.05,      # Scale between 90% and 110%
#     apply_phase=False,    # Not needed for real-valued images
#     apply_rotation=False,
#     apply_shift=False,
#     apply_scaling=False
# )


# Create TensorDataset instances
val_dataset = TensorDataset(val_data, val_labels, val_mask, Val_rel_x, Val_rel_y)
#train_dataset_unaugmented = TensorDataset(train_data, train_labels, train_mask)
test_dataset = TensorDataset(test_data, test_labels, test_mask, Test_rel_x, Test_rel_y)

# Create DataLoaders
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
##train_loader_noaugment = DataLoader(train_dataset_unaugmented, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

### train loader is treated separately due to data augmentations

aug_data, aug_labels, aug_mask, aug_rel_x, aug_rel_y = NN_input_train, ground_truth_train, Train_Mask, Train_rel_x, Train_rel_y         #transform_6d_data_labels_mask(NN_input_train, ground_truth_train, Train_Mask)
aug_mask = aug_mask + 1j*aug_mask #make sure it masks both channels real and imaginary
#aug_mask_gm = aug_mask_gm + 1j*aug_mask_gm #make sure it masks both channels real and imaginary
#aug_mask_wm = aug_mask_wm + 1j*aug_mask_wm #make sure it masks both channels real and imaginary

aug_data, aug_labels, aug_mask, aug_rel_x, aug_rel_y = aug_data.transpose(reshape_vector), aug_labels.transpose(reshape_vector), aug_mask.transpose(reshape_vector), aug_rel_x.transpose(reshape_vector), aug_rel_y.transpose(reshape_vector)

###collapse unused dimensions
aug_data = aug_data.reshape(Ground_Truth.shape[0], Ground_Truth.shape[1], Ground_Truth.shape[2], -1)
aug_labels = aug_labels.reshape(Ground_Truth.shape[0], Ground_Truth.shape[1], Ground_Truth.shape[2], -1)
aug_mask = aug_mask.reshape(Ground_Truth.shape[0], Ground_Truth.shape[1], Ground_Truth.shape[2], -1)
aug_rel_x = aug_rel_x.reshape(Ground_Truth.shape[0], Ground_Truth.shape[1], Ground_Truth.shape[2], -1)
aug_rel_y = aug_rel_y.reshape(Ground_Truth.shape[0], Ground_Truth.shape[1], Ground_Truth.shape[2], -1)


# normalized_input_train, normalized_ground_truth_train, norm_values_train = normalize_data_per_image_new(aug_data, aug_labels)
train_data, train_labels, train_mask, train_rel_x, train_rel_y  = reshape_for_pytorch(aug_data, grouped_time_steps), reshape_for_pytorch(aug_labels, grouped_time_steps), reshape_for_pytorch(aug_mask, grouped_time_steps), reshape_for_pytorch(aug_rel_x, grouped_time_steps, is_mask=True), reshape_for_pytorch(aug_rel_y, grouped_time_steps, is_mask=True)

train_dataset = TensorDataset(train_data, train_labels, train_mask, train_rel_x, train_rel_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

### !track unaugmented error! ###


# Prepare unaugmented training data (using the original training inputs, ground truth, and masks)
# train_mask_unaug = Train_Mask + 1j*Train_Mask #make sure it masks both channels real and imaginary

# train_data_unaug, train_labels_unaug, train_mask_unaug = NN_input_train.transpose(reshape_vector), ground_truth_train.transpose(reshape_vector), train_mask_unaug.transpose(reshape_vector)

# train_data_unaug = train_data_unaug.reshape(Ground_Truth.shape[0], Ground_Truth.shape[1], Ground_Truth.shape[2], -1)
# train_labels_unaug = train_labels_unaug.reshape(Ground_Truth.shape[0], Ground_Truth.shape[1], Ground_Truth.shape[2], -1)
# train_mask_unaug = train_mask_unaug.reshape(Ground_Truth.shape[0], Ground_Truth.shape[1], Ground_Truth.shape[2], -1)

# train_data_unaug = reshape_for_pytorch(train_data_unaug, grouped_time_steps)
# train_labels_unaug = reshape_for_pytorch(train_labels_unaug, grouped_time_steps)
# train_mask_unaug = reshape_for_pytorch(train_mask_unaug, grouped_time_steps)

# # Create a TensorDataset and DataLoader for the unaugmented data
#train_dataset_unaug = TensorDataset(train_data_unaug, train_labels_unaug, train_mask_unaug)
train_loader_unaug = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
#######

#### train loader treatment end

#### Initialize model ####

#### Initialize model ####

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StackedUNet3D(num_unets=Layers, use_batch_norm=False)  # or False if you don't want batch norm


#### Start training loop #####

# Define paths and configurations
latest_model_path = os.path.join(model_save_dir, 'latest_model.pth')
best_model_path   = os.path.join(model_save_dir, 'best_model.pth')

os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# ------------------------------------------------------------------------
#  Initialize model, optimizer, loss, etc.
# ------------------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.00002)

loss_fn = CustomLoss()  # Your custom loss

writer = SummaryWriter(log_dir=log_dir)

# Tracking lists for plotting/analysis (only MSE is tracked now)
train_mses = []
val_mses   = []
test_mses  = []

# Track start epoch & best validation MSE so far
start_epoch = 0
best_val_mse = float('inf')

# ------------------------------------------------------------------------
#  If a "latest model" checkpoint exists, resume training from there
# ------------------------------------------------------------------------
if os.path.exists(latest_model_path):
    print(f"Found existing model at {latest_model_path}. Resuming training...")
    checkpoint = torch.load(latest_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
    
    start_epoch    = checkpoint['epoch']
    train_mses     = checkpoint.get('train_mses', [])
    val_mses       = checkpoint.get('val_mses', [])
    test_mses      = checkpoint.get('test_mses', [])
    best_val_mse   = checkpoint.get('best_val_mse', float('inf'))

model = model.to(device)

# ------------------------------------------------------------------------
#  Main training loop
# ------------------------------------------------------------------------
for epoch in range(start_epoch, num_epochs):
    # ---- Train for one epoch ----
    _ = train_one_epoch(model, optimizer, loss_fn, train_loader, device=device)
    
    # ---- Evaluate on TRAIN data ----
    avg_loss_train = validate_model(model, loss_fn, train_loader_unaug, device=device)
    
    # ---- Evaluate on VALIDATION data ----
    avg_loss_val = validate_model(model, loss_fn, val_loader, device=device)
    
    # ---- Evaluate on TEST data ----
    avg_loss_test = validate_model(model, loss_fn, test_loader, device=device)
    
    # ---- Update tracked lists ----
    train_mses.append(avg_loss_train)
    val_mses.append(avg_loss_val)
    test_mses.append(avg_loss_test)
    
    # ---- TensorBoard Logging ----
    writer.add_scalar('Loss/Train', avg_loss_train, epoch)
    writer.add_scalar('Loss/Validation', avg_loss_val, epoch)
    writer.add_scalar('Loss/Test', avg_loss_test, epoch)
    
    # ---- Always save the "latest" model checkpoint ----
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_mses': train_mses,
        'val_mses': val_mses,
        'test_mses': test_mses,
        'best_val_mse': best_val_mse
    }, latest_model_path)
    
    # ---- If we found a new "best" model on the VALIDATION set, save separately ----
    if avg_loss_val < best_val_mse:
        best_val_mse = avg_loss_val
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_mses': train_mses,
            'val_mses': val_mses,
            'test_mses': test_mses,
            'best_val_mse': best_val_mse
        }, best_model_path)
        print(f"New best model found at epoch {epoch+1} with Validation MSE = {avg_loss_val:.6f}")
        print(f"Saved to {best_model_path}\n")
    
    # ---- Print progress every 'print_every' epochs ----
    if (epoch + 1) % print_every == 0:
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"   Train Loss:      {avg_loss_train:.6f}")
        print(f"   Validation Loss: {avg_loss_val:.6f}")
        print(f"   Test Loss:       {avg_loss_test:.6f}\n")
        
    ### train loader is treated separately due to data augmentations, RESET!

    aug_data, aug_labels, aug_mask, aug_rel_x, aug_rel_y = transform_6d_data_labels_mask(NN_input_train, ground_truth_train, Train_Mask, Train_rel_x, Train_rel_y)         #transform_6d_data_labels_mask(NN_input_train, ground_truth_train, Train_Mask)
    aug_mask = aug_mask + 1j*aug_mask #make sure it masks both channels real and imaginary
    #aug_mask_gm = aug_mask_gm + 1j*aug_mask_gm #make sure it masks both channels real and imaginary
    #aug_mask_wm = aug_mask_wm + 1j*aug_mask_wm #make sure it masks both channels real and imaginary

    aug_data, aug_labels, aug_mask, aug_rel_x, aug_rel_y = aug_data.transpose(reshape_vector), aug_labels.transpose(reshape_vector), aug_mask.transpose(reshape_vector), aug_rel_x.transpose(reshape_vector), aug_rel_y.transpose(reshape_vector)

    ###collapse unused dimensions
    aug_data = aug_data.reshape(Ground_Truth.shape[0], Ground_Truth.shape[1], Ground_Truth.shape[2], -1)
    aug_labels = aug_labels.reshape(Ground_Truth.shape[0], Ground_Truth.shape[1], Ground_Truth.shape[2], -1)
    aug_mask = aug_mask.reshape(Ground_Truth.shape[0], Ground_Truth.shape[1], Ground_Truth.shape[2], -1)
    aug_rel_x = aug_rel_x.reshape(Ground_Truth.shape[0], Ground_Truth.shape[1], Ground_Truth.shape[2], -1)
    aug_rel_y = aug_rel_y.reshape(Ground_Truth.shape[0], Ground_Truth.shape[1], Ground_Truth.shape[2], -1)


    # normalized_input_train, normalized_ground_truth_train, norm_values_train = normalize_data_per_image_new(aug_data, aug_labels)
    train_data, train_labels, train_mask, train_rel_x, train_rel_y  = reshape_for_pytorch(aug_data, grouped_time_steps), reshape_for_pytorch(aug_labels, grouped_time_steps), reshape_for_pytorch(aug_mask, grouped_time_steps), reshape_for_pytorch(aug_rel_x, grouped_time_steps, is_mask=True), reshape_for_pytorch(aug_rel_y, grouped_time_steps, is_mask=True)

    train_dataset = TensorDataset(train_data, train_labels, train_mask, train_rel_x, train_rel_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#### train loader treatment end

print("Training complete.")
writer.close()








    
