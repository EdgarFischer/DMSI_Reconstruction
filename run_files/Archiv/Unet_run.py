import os
import sys

sys.path.append('../scripts')
sys.path.append('../models')

os.environ["CUDA_VISIBLE_DEVICES"]= '2' #, this way I would choose GPU 3 to do the work

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom # for compressing images / only for testing purposes to speed up NN training
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch.nn as nn
from data_preparation import *
from data_undersampling import *
from Unet import *
from output_statistics import *

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio 

grouped_time_steps = 1 # Set how many subsequent time steps you want to give to the network at once. Values allowed: 1, 2, 4, 8 (because it has to divide 8)

######## SET PARAMETERS ########
#### Undersampling Strategy:#####
Undersampling = "Regular" # Options: Regular or Possoin
Sampling_Mask = "One_Mask" #Options: Single_Combination or One_Mask or Complementary_Masks
AF = 2 #  acceleration factor

#### Model Input and Output ####
GT_Data = "LowRank" # Options: FullRank LowRank for GROUNDTRUTH! 
Low_Rank_Input = True ## apply low rank to the input as well if True

batch_size=128
num_epochs = 500
print_every = 1
trancuate_t = 96

#### Parameter setting END ####

#### Define ground truth path####
if GT_Data == "FullRank":
    ground_truth_path = "../data/Ground_Truth/Full_Rank/P03-P08_truncated_k_space.npy"
elif GT_Data == "LowRank":
    ground_truth_path = "../data/Ground_Truth/Low_Rank/LR_8_P03-P08_self.npy"

#### Assemble saving path ####

#### Definie Model path
if GT_Data == "FullRank":
    model_save_dir = f"../saved_models/UNet/Full2Full/"+Undersampling+f'/AF_{AF}/'+f'Truncate_t_{trancuate_t}/'+Sampling_Mask
elif GT_Data == "LowRank":
    model_save_dir = f"../saved_models/UNet/Low2Low/"+Undersampling+f'/AF_{AF}/'+f'Truncate_t_{trancuate_t}/'+Sampling_Mask
    
#### Define log directory path
if GT_Data == "FullRank":
    log_dir = f"../log_files/UNet/Full2Full/"+Undersampling+f'/AF_{AF}/'+f'Truncate_t_{trancuate_t}/'+Sampling_Mask
elif GT_Data == "LowRank":
    log_dir = f"../log_files/UNet/Low2Low/"+Undersampling+f'/AF_{AF}/'+f'Truncate_t_{trancuate_t}/'+Sampling_Mask

#### Define Input Data path
undersampled_data_path = "../data/Undersampled_Data/"+Undersampling+f'/AF_{AF}/'+Sampling_Mask+'/data.npy'

#### load data!
Ground_Truth = np.load(ground_truth_path)
Undersampled_Data = np.load(undersampled_data_path)
MASKS = np.load("../data/masks.npy")

####
mask_expanded = MASKS[:, :, :, None, None, :]  # Now shape is (22,22,21,1,1,6)
# Use broadcasting to "repeat" the mask along these new axes:
mask_extended = np.broadcast_to(mask_expanded, (22, 22, 21, 96, 8, 6))
mask_extended = mask_extended + 1J*mask_extended

#### additionally make LowRank 8 transformation on input of network, this improves the error significantly!
if Low_Rank_Input:
    Undersampled_Data[...,0] = low_rank(Undersampled_Data[...,0], 8)
    Undersampled_Data[...,1] = low_rank(Undersampled_Data[...,1], 8)
    Undersampled_Data[...,2] = low_rank(Undersampled_Data[...,2], 8)
    Undersampled_Data[...,3] = low_rank(Undersampled_Data[...,3], 8)
    Undersampled_Data[...,4] = low_rank(Undersampled_Data[...,4], 8)
    Undersampled_Data[...,5] = low_rank(Undersampled_Data[...,5], 8)

#### Data transformations !
####### TEEEEST!!!! ##########
# Create three dummy tensors of shape (22,22,22,96,8,6)

# # Generate the same random permutations for all tensors
# perm_4 = np.random.permutation(96)  # 4th dim (96)
# perm_5 = np.random.permutation(8)  # 5th dim (8)
# perm_6 = np.random.permutation(6)  # 6th dim (6)

# # Apply the same permutations to all three tensors
# Ground_Truth = Ground_Truth[:, :, :, perm_4, :, :]
# Ground_Truth = Ground_Truth[:, :, :, :, perm_5, :]
# Ground_Truth = Ground_Truth[:, :, :, :, :, perm_6]

# Undersampled_Data = Undersampled_Data[:, :, :, perm_4, :, :]
# Undersampled_Data = Undersampled_Data[:, :, :, :, perm_5, :]
# Undersampled_Data = Undersampled_Data[:, :, :, :, :, perm_6]

# mask_extended = mask_extended[:, :, :, perm_4, :, :]
# mask_extended = mask_extended[:, :, :, :, perm_5, :]
# mask_extended = mask_extended[:, :, :, :, :, perm_6]

##### test ende ####
ground_truth_train, ground_truth_test = Ground_Truth[:,:,:,:trancuate_t,:,1:6], Ground_Truth[:,:,:,:trancuate_t,:,0]  # Method: Leave last MRSI measurement as test set
Train_Mask, Test_Mask = mask_extended[:,:,:,:trancuate_t,:,1:6], mask_extended[:,:,:,:trancuate_t,:,0]

#### Assign undersampled network input ####
NN_input_train, NN_input_test = Undersampled_Data[:,:,:,:trancuate_t,:,1:6], Undersampled_Data[:,:,:,:trancuate_t,:,0]

#### Collapse ununsed dimensions ####
ground_truth_train, ground_truth_test = ground_truth_train.reshape(22, 22, 21, -1), ground_truth_test.reshape(22, 22, 21, -1)
NN_input_train, NN_input_test = NN_input_train.reshape(22, 22, 21, -1), NN_input_test.reshape(22, 22, 21, -1)
Mask_train, Mask_test = Train_Mask.reshape(22, 22, 21, -1), Test_Mask.reshape(22, 22, 21, -1)

#### Normalize data #####
normalized_input_train, normalized_ground_truth_train, norm_values_train = normalize_data_per_image_new(NN_input_train, ground_truth_train)
normalized_input_test, normalized_ground_truth_test, norm_values_test = normalize_data_per_image_new(NN_input_test, ground_truth_test)

#### reshape for pytorch ####
train_data, train_labels  = reshape_for_pytorch(normalized_input_train, grouped_time_steps), reshape_for_pytorch(normalized_ground_truth_train, grouped_time_steps)
test_data, test_labels = reshape_for_pytorch(normalized_input_test, grouped_time_steps), reshape_for_pytorch(normalized_ground_truth_test, grouped_time_steps)
train_mask, test_mask = reshape_for_pytorch(Mask_train, grouped_time_steps), reshape_for_pytorch(Mask_test, grouped_time_steps)

#### increase dimensions to 24x24x24 for Unet
pad_width = ((0, 0), (0, 0), (1, 1), (1, 1), (1, 2))
train_data, train_labels = np.pad(train_data, pad_width, mode='constant', constant_values=0), np.pad(train_labels, pad_width, mode='constant', constant_values=0)
test_data, test_labels = np.pad(test_data, pad_width, mode='constant', constant_values=0), np.pad(test_labels, pad_width, mode='constant', constant_values=0)
train_mask, test_mask = np.pad(train_mask, pad_width, mode='constant', constant_values=0), np.pad(test_mask, pad_width, mode='constant', constant_values=0)
#### prepare data for NN #####


# Import the augmentation class
augment = RandomAugment3D(
    rotation_range=0.3,   # Rotate by up to ±17° in xy and xz planes
    shift_pixels=1,       # Shift up to 2 pixels in any direction
    scale_range=0.05,      # Scale between 90% and 110%
    apply_phase=False,    # Not needed for real-valued images
    apply_rotation=True,
    apply_shift=True,
    apply_scaling=True
)


# Create TensorDataset instances
train_dataset = TensorDatasetWithAugmentation(train_data, train_labels, train_mask, norm_values_train, transform=augment)
train_dataset_unaugmented = TensorDatasetWithAugmentation(train_data, train_labels, train_mask, norm_values_train, transform=None)
test_dataset = TensorDatasetWithAugmentation(test_data, test_labels, test_mask, norm_values_test, transform=None)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_loader_noaugment = DataLoader(train_dataset_unaugmented, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#### Initialize model ####

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet3D().to(device)


#### Start training loop #####

# Define paths and configurations

latest_model_path = os.path.join(model_save_dir, 'latest_model.pth')
best_model_path   = os.path.join(model_save_dir, 'best_model.pth')

os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# ------------------------------------------------------------------------
#  Initialize model, optimizer, loss, metrics, etc.
# ------------------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.00002)
loss_fn = CustomLoss()  # Your custom loss
psnr_metric = PeakSignalNoiseRatio(data_range=2.0).to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)

writer = SummaryWriter(log_dir=log_dir)

# Tracking lists for plotting/analysis
train_mses = []
train_psnrs = []
train_ssims = []
test_mses = []
test_psnrs = []
test_ssims = []

# Track start epoch & best MSE so far
start_epoch = 0
best_test_mse = float('inf')  # We'll update this if we find a new best

# ------------------------------------------------------------------------
#  If a "latest model" checkpoint exists, resume training from there
# ------------------------------------------------------------------------
if os.path.exists(latest_model_path):
    print(f"Found existing model at {latest_model_path}. Resuming training...")
    checkpoint = torch.load(latest_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch     = checkpoint['epoch']
    train_mses      = checkpoint.get('train_mses', [])
    test_mses       = checkpoint.get('test_mses', [])
    train_psnrs     = checkpoint.get('train_psnrs', [])
    test_psnrs      = checkpoint.get('test_psnrs', [])
    train_ssims     = checkpoint.get('train_ssims', [])
    test_ssims      = checkpoint.get('test_ssims', [])
    best_test_mse   = checkpoint.get('best_test_mse', float('inf'))

model = model.to(device)

# ------------------------------------------------------------------------
#  Main training loop
# ------------------------------------------------------------------------
for epoch in range(start_epoch, num_epochs):
    # ---- Train for one epoch ----
    _ = train_one_epoch(model, optimizer, loss_fn, train_loader, device=device)
    
    # ---- Evaluate on TRAIN data ----
    psnr_metric.reset()
    ssim_metric.reset()
    avg_loss_train, avg_psnr_train, avg_ssim_train = validate_model(
        model, loss_fn, train_loader_noaugment, device=device,
        psnr_metric=psnr_metric,
        ssim_metric=ssim_metric
    )
    
    # ---- Evaluate on TEST data ----
    psnr_metric.reset()
    ssim_metric.reset()
    avg_loss_test, avg_psnr_test, avg_ssim_test = validate_model(
        model, loss_fn, test_loader, device=device,
        psnr_metric=psnr_metric,
        ssim_metric=ssim_metric
    )
    
    # ---- (Optional) Scheduler step ----
    # scheduler.step(avg_loss_train)
    
    # ---- Update tracked lists ----
    train_mses.append(avg_loss_train)
    train_psnrs.append(avg_psnr_train)
    train_ssims.append(avg_ssim_train)

    test_mses.append(avg_loss_test)
    test_psnrs.append(avg_psnr_test)
    test_ssims.append(avg_ssim_test)
    
    # ---- TensorBoard Logging ----
    writer.add_scalar('Loss/Train',        avg_loss_train, epoch)
    writer.add_scalar('Loss/Test',         avg_loss_test,  epoch)
    writer.add_scalar('Metric/PSNR /Train',avg_psnr_train, epoch)
    writer.add_scalar('Metric/PSNR /Test', avg_psnr_test,  epoch)
    writer.add_scalar('Metric/SSIM /Train',avg_ssim_train, epoch)
    writer.add_scalar('Metric/SSIM /Test', avg_ssim_test,  epoch)
    
    # ---- Always save the "latest" model checkpoint ----
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_mses': train_mses,
        'test_mses': test_mses,
        'train_psnrs': train_psnrs,
        'test_psnrs': test_psnrs,
        'train_ssims': train_ssims,
        'test_ssims': test_ssims,
        # Keep track of the current best test MSE in this checkpoint too
        'best_test_mse': best_test_mse
    }, latest_model_path)
    
    # ---- If we found a new "best" model on the TEST set, save separately ----
    if avg_loss_test < best_test_mse:
        best_test_mse = avg_loss_test
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_mses': train_mses,
            'test_mses': test_mses,
            'train_psnrs': train_psnrs,
            'test_psnrs': test_psnrs,
            'train_ssims': train_ssims,
            'test_ssims': test_ssims,
            'best_test_mse': best_test_mse
        }, best_model_path)
        print(f"New best model found at epoch {epoch+1} with Test MSE = {avg_loss_test:.6f}")
        print(f"Saved to {best_model_path}\n")

    # ---- Print progress every 'print_every' epochs ----
    if (epoch + 1) % print_every == 0:
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"   Train Loss: {avg_loss_train:.6f}")
        print(f"   Test  Loss: {avg_loss_test:.6f}")
        print(f"   Train PSNR: {avg_psnr_train:.4f}")
        print(f"   Test  PSNR: {avg_psnr_test:.4f}")
        print(f"   Train SSIM: {avg_ssim_train:.4f}")
        print(f"   Test  SSIM: {avg_ssim_test:.4f}\n")

print("Training complete.")
writer.close()







    
