# This file is used to configure a 3D Unet architecture

CUDA = '1'  # choose which GPU to use for training

#### Data specification ####
# NOTE:  not all options listed may be available, you have to check that the corresponding data for given acceleration factor
# has actually been computed


Undersampling = "Possoin_Real"                  # Options: Regular or Possoin, Regular_Real, Possoin_Real
Sampling_Mask = "Complementary_Masks"           # Options: Single_Combination or One_Mask or Complementary_Masks
AF = 4                                          # acceleration factor
DOMAIN = "zfT"                                  # Input axes that goes into the network (as well as output), 
                                                # valid options: kzfT (this means k_z in kspace); zfT; ztT; xyz; xzT; xyf
dataset = 'data_small_matrix.npy'               # name of dataset for training
training_set = [0]                              # choose subject index for training set
test_set = [0]                                  # choose subject index (assumed to be last index) for testing
test_frac = 0.2                                 # NOTE: This option is only active if test_set = training_set and will otherwise be 
                                                # ignored. In this a fraction of the same data set is used for testing

# Zero padding can be used to ensure that total number of z slices for example is divisible by 8, which is necessary for the Unet
zero_pad_xyz = 40
zero_pad_t = 112
zero_pad_T = 8
trancuate_t = zero_pad_t

#### Model Input and Output ####
GT_Data = "LowRank"                             # Options: FullRank LowRank for GROUNDTRUTH! 
Low_Rank_Input = False                          # apply low rank to the input as well if True
grouped_time_steps = 1                          # you can group time steps in the channel dimension, 

#### Trainign parameters ####
batch_size=120
num_epochs = 2000
print_every = 1                                 # print statistics after given number of epochs

#### Model and log folder names ####
folder_name = 'Small_Matrix_1.1'



