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
from Naive_CNN_3D import *
from output_statistics import *

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio 

grouped_time_steps = 1 # Set how many subsequent time steps you want to give to the network at once. Values allowed: 1, 2, 4, 8 (because it has to divide 8)

######## SET PARAMETERS ########

batch_size=64
num_convs = 10
num_epochs = 1000
print_every = 1
trancuate_t = 96
SC = True # SC = apply complementary masks

r=0 ### fixed radius possoin

if SC:
    model_save_dir = f'../saved_models/x-z-T-Naive_CNN_3D_Regular_Full2Low_SC_{trancuate_t}/{num_convs}Layer'
    log_dir = f'../log_files/x-z-T-Naive_CNN_3D_Regular_Full2Low_SC_{trancuate_t}/{num_convs}Layer'
    Undersampled_Data = np.load(f'../data/Undersampled_Data_P03-P08/Undersampled_Regular_complementary_masks.npy')
else:
    model_save_dir = f'../saved_models/x-z-T-Naive_CNN_3D_Regular_Full2Low_{trancuate_t}/{num_convs}Layer'
    log_dir = f'../log_files/x-z-T-Naive_CNN_3D_Regular_Full2Low_{trancuate_t}/{num_convs}Layer'
    Undersampled_Data = np.load(f'../data/Undersampled_Data_P03-P08/Undersampled_Regular.npy')
################################

#### load data
Ground_Truth = np.load('../data/LR_8_P03-P08_self.npy')

#### additionally make LowRank 8 transformation on input of network, this improves the error significantly!
Undersampled_Data[...,0] = low_rank(Undersampled_Data[...,0], 8)
Undersampled_Data[...,1] = low_rank(Undersampled_Data[...,1], 8)
Undersampled_Data[...,2] = low_rank(Undersampled_Data[...,2], 8)
Undersampled_Data[...,3] = low_rank(Undersampled_Data[...,3], 8)
Undersampled_Data[...,4] = low_rank(Undersampled_Data[...,4], 8)
Undersampled_Data[...,5] = low_rank(Undersampled_Data[...,5], 8)

#### Transpose for x-z-T network
Ground_Truth = np.transpose(Ground_Truth, (0, 2, 4, 3, 1, 5))
Undersampled_Data = np.transpose(Undersampled_Data, (0, 2, 4, 3, 1, 5))


 ## Data set with accerleration factor 3

ground_truth_train, ground_truth_test = Ground_Truth[:,:,:,:trancuate_t,:,:5], Ground_Truth[:,:,:,:trancuate_t,:,5]  # Method: Leave last MRSI measurement as test set

#### Assign undersampled network input ####
NN_input_train, NN_input_test = Undersampled_Data[:,:,:,:trancuate_t,:,:5], Undersampled_Data[:,:,:,:trancuate_t,:,5]

#### Collapse ununsed dimensions ####
ground_truth_train, ground_truth_test = ground_truth_train.reshape(22, 21, 8, -1), ground_truth_test.reshape(22, 21, 8, -1)
NN_input_train, NN_input_test = NN_input_train.reshape(22, 21, 8, -1), NN_input_test.reshape(22, 21, 8, -1)

#### Normalize data #####
normalized_input_train, normalized_ground_truth_train, _ = normalize_data_per_image_new(NN_input_train, ground_truth_train)
normalized_input_test, normalized_ground_truth_test, _ = normalize_data_per_image_new(NN_input_test, ground_truth_test)

#### reshape for pytorch ####
train_data, train_labels  = reshape_for_pytorch(normalized_input_train, grouped_time_steps), reshape_for_pytorch(normalized_ground_truth_train, grouped_time_steps)
test_data, test_labels = reshape_for_pytorch(normalized_input_test, grouped_time_steps), reshape_for_pytorch(normalized_ground_truth_test, grouped_time_steps)

#### prepare data for NN #####

# Create TensorDataset instances
train_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#### Initialize model ####

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Naive_CNN_3D(grouped_time_steps=grouped_time_steps, num_convs=num_convs).to(device)


#### Start training loop #####

# Define paths and configurations

model_save_path = os.path.join(model_save_dir, 'model.pth')

os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Initialize model, optimizer, and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.00002)
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
loss_fn = CustomLoss()

# Set up metrics
psnr_metric = PeakSignalNoiseRatio(data_range=2.0).to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)

# TensorBoard writer
writer = SummaryWriter(log_dir=log_dir)

# Variables to track training progress
start_epoch = 0
train_mses = []
train_psnrs = []
train_ssims = []
test_mses = []
test_psnrs = []
test_ssims = []

# Check if a saved model exists
if os.path.exists(model_save_path):
    print(f"Found existing model at {model_save_path}. Resuming training...")
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    train_mses = checkpoint.get('train_mses', [])
    test_mses = checkpoint.get('test_mses', [])
    train_psnrs = checkpoint.get('train_psnrs', [])
    test_psnrs = checkpoint.get('test_psnrs', [])
    train_ssims = checkpoint.get('train_ssims', [])
    test_ssims = checkpoint.get('test_ssims', [])

model = model.to(device)

for epoch in range(start_epoch, num_epochs):
    _ = train_one_epoch(model, optimizer, loss_fn, train_loader, device=device)

    psnr_metric.reset()
    ssim_metric.reset()
    
    avg_loss_train, avg_psnr_train, avg_ssim_train = validate_model(
            model, loss_fn, train_loader, device=device,
            psnr_metric=psnr_metric,
            ssim_metric=ssim_metric
        )
    
    psnr_metric.reset()
    ssim_metric.reset()
    avg_loss_test, avg_psnr_test, avg_ssim_test = validate_model(
            model, loss_fn, test_loader, device=device,
            psnr_metric=psnr_metric,
            ssim_metric=ssim_metric
        )
    
    #scheduler.step(avg_loss_train)
    
    train_mses.append(avg_loss_train)
    train_psnrs.append(avg_psnr_train)
    train_ssims.append(avg_ssim_train)
    
    test_mses.append(avg_loss_test)
    test_psnrs.append(avg_psnr_test)
    test_ssims.append(avg_ssim_test)
    
    
    
    
    
    writer.add_scalar('Loss/Train', avg_loss_train, epoch)
    writer.add_scalar('Loss/Test', avg_loss_test, epoch)
    writer.add_scalar('Metric/PSNR /Train', avg_psnr_train, epoch)
    writer.add_scalar('Metric/PSNR /Test', avg_psnr_test, epoch)
    writer.add_scalar('Metric/SSIM /Train', avg_ssim_train, epoch)
    writer.add_scalar('Metric/SSIM /Test', avg_ssim_test, epoch)
    
    psnr_metric.reset()
    ssim_metric.reset()
    
    # Save the model at the end of every epoch
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_mses': train_mses,
        'test_mses': test_mses,
        'train_psnrs': train_psnrs,
        'test_psnrs': test_psnrs,
        'train_ssims': train_ssims,
        'test_ssims': test_ssims
    }, model_save_path)
    
    if (epoch + 1) % print_every == 0:
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"   Train Loss: {avg_loss_train:.6f}")
        print(f"   Test  Loss: {avg_loss_test:.6f}")
        print(f"   Train  PSNR: {avg_psnr_train:.4f}")
        print(f"   Test  PSNR: {avg_psnr_test:.4f}")
        print(f"   Train  SSIM: {avg_ssim_train:.4f}")
        print(f"   Test  SSIM: {avg_ssim_test:.4f}\n")







    
