from scipy.io import loadmat
import numpy as np
import math
import random

def apply_undersampling(data, strategy,  undersampling_factor, grouped_time_steps, fixed_radius = 5, T_Combination = False, r_max = 10):
    """
    Master function to apply an undersampling strategy with grouping.

    Args:
        data (np.ndarray): Input dataset with shape (N_x, N_y, N_z, t, grouped_time_steps, ...).
        strategy (str): Keyword for the undersampling strategy ('uniform', etc.).
        factor (float): Undersampling factor (0 < factor <= 1).
        grouped_time_steps (int): Number of time steps to group for the masks.
        fixed_radius (float): Radius of the fixed sphere for undersampling.
        T_Combination: If complementary undersampling is chosen for big T, fill k-spaces 0 from next time step
        sampled_mask: Masks that is 1 for k-space points that have been sampled in the ground truth (sphere)
        r_max: only relevant for possoin
        
    Returns:
        np.ndarray: Undersampled dataset with the same shape as input data.
    """
    strategies = {
        "uniform": generate_uniform_masks,
        "uniform_complementary": generate_complementary_uniform_masks,
        "possoin": create_possoin_undersampling_mask
    }

    if strategy not in strategies:
        raise ValueError(f"Unknown undersampling strategy: {strategy}")
        
    if strategy == "possoin":
        if not grouped_time_steps == 1:
            raise ValueError(f"Only grouped_time_steps = 1 supported: {strategy}")
        
        shape = (22,22,21) ####!!!! May later have to be fixed !!!!!#####
        sampled_mask = fully_sampled_mask(r_max,shape)
        masks = [create_possoin_undersampling_mask(fixed_radius, r_max, 1/undersampling_factor, sampled_mask)] 
        
    else: 
        masks = strategies[strategy](data.shape[:3], fixed_radius, undersampling_factor, grouped_time_steps)
        
    # Apply the masks to the data
    undersampled_data = np.zeros_like(data)  # Initialize undersampled data with same shape
    
    if grouped_time_steps == 1:
        mask = masks[0]
        
        # Dynamically expand mask to match data's trailing dimensions
        num_trailing_dims = data.ndim - mask.ndim  # Number of trailing dimensions
        mask_expanded = mask.reshape(mask.shape + (1,) * num_trailing_dims)  # Dynamically expand
        
        # Apply the mask
        undersampled_data[:,:,:, ...] = data[:,:,:, ...] * mask_expanded
        
        return undersampled_data
    
    for time_index in range(grouped_time_steps):
    # Get the mask for the current time index
        mask = masks[time_index]

        # Dynamically expand mask to match data's trailing dimensions
        num_trailing_dims = data.ndim - mask.ndim - 1  # Number of trailing dimensions
        mask_expanded = mask.reshape(mask.shape + (1,) * num_trailing_dims)  # Dynamically expand

        # Apply the mask
        undersampled_data[:,:,:,:, time_index, ...] = data[:,:,:,:, time_index, ...] * mask_expanded
        
    

    return undersampled_data    

def generate_uniform_masks(shape, fixed_radius, undersampling_factor, grouped_time_steps):
    """
    Generate 3D Cartesian k-space undersampling masks with a fixed sphere and uniform random sampling outside.

    Args:
        shape (tuple): Shape of the k-space (nx, ny, nz).
        fixed_radius (float): Radius of the central sphere to preserve.
        undersampling_factor (float): Uniform random sampling factor outside the fixed sphere.
        grouped_time_steps (int): Number of time steps (masks to generate).

    Returns:
        list[np.ndarray]: A list of binary masks (1 = sample, 0 = skip), one for each time step.
    """
    if not (0 < undersampling_factor <= 1):
        raise ValueError("The undersampling factor must be in the interval (0, 1].")
    
    nx, ny, nz = shape
    center = np.array([nx // 2, ny // 2, nz // 2])

    # Create a grid for distance calculation
    x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    distances = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)

    # Fixed sphere: Keep all points inside the sphere
    fixed_sphere_mask = distances <= fixed_radius
    
    # Generate uniform random sampling for points outside the sphere
    outside_sphere_mask = ~fixed_sphere_mask
    outside_indices = np.where(outside_sphere_mask)

    # Create masks for each time step
    masks = []
    for time_index in range(grouped_time_steps):
        # Random seed for reproducibility and different patterns for each time step
        np.random.seed(time_index)

        # Randomly select points to keep outside the sphere
        num_points_to_keep = int(len(outside_indices[0]) * undersampling_factor)
        selected_indices = np.random.choice(
            len(outside_indices[0]),
            size=num_points_to_keep,
            replace=False,
        )
        sampled_coords = tuple(idx[selected_indices] for idx in outside_indices)

        # Create the mask
        mask = fixed_sphere_mask.copy()
        mask[sampled_coords] = True  # Add random points outside the sphere
        masks.append(mask)
        
        print("Real undersampling factor")
        print(np.sum(masks[0])/(4169))

    return masks

def generate_complementary_uniform_masks(shape, fixed_radius, undersampling_factor, grouped_time_steps):
    """
    Generate 3D Cartesian k-space undersampling masks with a fixed sphere and complementary uniform random sampling outside.

    Args:
        shape (tuple): Shape of the k-space (nx, ny, nz).
        fixed_radius (float): Radius of the central sphere to preserve.
        undersampling_factor (float): Uniform random sampling factor outside the fixed sphere.
        grouped_time_steps (int): Number of time steps (masks to generate).

    Returns:
        list[np.ndarray]: A list of binary masks (1 = sample, 0 = skip), one for each time step.
    """
    if not (0 < undersampling_factor <= 1):
        raise ValueError("The undersampling factor must be in the interval (0, 1].")

    nx, ny, nz = shape
    center = np.array([nx // 2, ny // 2, nz // 2])

    # Create a grid for distance calculation
    x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    distances = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)

    # Fixed sphere: Keep all points inside the sphere
    fixed_sphere_mask = distances <= fixed_radius

    # Generate indices outside the sphere
    outside_sphere_mask = ~fixed_sphere_mask
    outside_indices = np.where(outside_sphere_mask)

    # Calculate the total number of points to keep outside the fixed sphere
    num_points_to_keep = int(len(outside_indices[0]) * undersampling_factor)

    # Create an array to track sampled points
    previously_sampled = np.zeros_like(fixed_sphere_mask, dtype=bool)

    # Initialize the list of masks
    masks = []

    for time_index in range(grouped_time_steps):
        # Identify unsampled points outside the sphere
        unsampled_indices = np.where(outside_sphere_mask & ~previously_sampled)

        # If there are not enough unsampled points left, allow random resampling
        if len(unsampled_indices[0]) < num_points_to_keep:
            available_indices = np.where(outside_sphere_mask)
        else:
            available_indices = unsampled_indices

        # Randomly select points to keep outside the sphere
        selected_indices = np.random.choice(
            len(available_indices[0]),
            size=min(num_points_to_keep, len(available_indices[0])),
            replace=False,
        )
        sampled_coords = tuple(idx[selected_indices] for idx in available_indices)

        # Create the mask
        mask = fixed_sphere_mask.copy()
        mask[sampled_coords] = True  # Add random points outside the sphere

        # Update the previously sampled points
        previously_sampled[sampled_coords] = True

        # Append the mask to the list
        masks.append(mask.copy())
        
        print("real undersampling factor")
        print(np.sum(masks[0]/(4169)))

    return masks

def two_masks_undersampling(training_images, test_images, fixed_radius, undersampling_factor):
    from data_preparation import fourier_transform
    training_FT = fourier_transform(training_images)
    test_FT = fourier_transform(test_images)

        # -------------------------
        # 3) Undersample in k-space
        # -------------------------


    #### next I undersample manually    

    masks = generate_complementary_uniform_masks(training_FT.shape[:3], fixed_radius, undersampling_factor, 2) 

    for i in range(2):
        masks[i] = masks[i].reshape(masks[i].shape + (1,) * 1)
    #mask_1 = masks[1].reshape(masks[1].shape + (1,) * 1)

    test_undersampled = np.zeros_like(test_FT)

    for i in range(test_FT.shape[-1]):
        if i % 2 == 0:
            test_undersampled[...,i] = test_FT[...,i] * masks[0]
        else:
            test_undersampled[...,i] = test_FT[...,i] * masks[1]

    for i in range(2):
        masks[i] = masks[i].reshape(masks[i].shape + (1,) * 1)

    training_undersampled = np.zeros_like(training_FT)

    for i in range(test_FT.shape[-1]):
        if i % 2 == 0:
            training_undersampled[...,i,:] = training_FT[...,i,:] * masks[0]
        else:
            training_undersampled[...,i,:] = training_FT[...,i,:] * masks[1]


    #check which entries are 0
    mask_0 = np.where(training_undersampled[:,:,:,:,0,...] != 0, 0, 1)
    mask_1 = np.where(training_undersampled[:,:,:,:,1,...] != 0, 0, 1)

    training_undersampled[:,:,:,:,0,...] = training_undersampled[:,:,:,:,0,...] + training_undersampled[:,:,:,:,1,...]*mask_0

    for i in range(1,7):
        if i % 2 == 0:
            training_undersampled[:,:,:,:,i,...] = training_undersampled[:,:,:,:,i,...] + 0.5*(training_undersampled[:,:,:,:,i+1,...]*mask_0+training_undersampled[:,:,:,:,i-1,...]*mask_0)
        else:
            training_undersampled[:,:,:,:,i,...] = training_undersampled[:,:,:,:,i,...] + 0.5*(training_undersampled[:,:,:,:,i+1,...]*mask_0+training_undersampled[:,:,:,:,i-1,...]*mask_1)

    training_undersampled[:,:,:,:,7,...] = training_undersampled[:,:,:,:,7,...] + training_undersampled[:,:,:,:,6,...]*mask_1    

    #check which entries are 0
    mask_0 = np.where(test_undersampled[:,:,:,:,0,...] != 0, 0, 1)
    mask_1 = np.where(test_undersampled[:,:,:,:,1,...] != 0, 0, 1)

    test_undersampled[:,:,:,:,0,...] = test_undersampled[:,:,:,:,0,...] + test_undersampled[:,:,:,:,1,...]*mask_0

    for i in range(1,7):
        if i % 2 == 0:
            test_undersampled[:,:,:,:,i,...] = test_undersampled[:,:,:,:,i,...] + 0.5*(test_undersampled[:,:,:,:,i+1,...]*mask_0+test_undersampled[:,:,:,:,i-1,...]*mask_0)
        else:
            test_undersampled[:,:,:,:,i,...] = test_undersampled[:,:,:,:,i,...] + 0.5*(test_undersampled[:,:,:,:,i+1,...]*mask_0+test_undersampled[:,:,:,:,i-1,...]*mask_1)

    test_undersampled[:,:,:,:,7,...] = test_undersampled[:,:,:,:,7,...] + test_undersampled[:,:,:,:,6,...]*mask_1
    
    return training_undersampled, test_undersampled

def generate_random_possoin_indices(r_min, r_max, shape):
    nx, ny, nz = shape
    center = np.array([nx // 2, ny // 2, nz // 2])
    
    #generate vector  in spherical coordinates, wrt to center as origin
    r = random.uniform(r_min, r_max) ## note that this gives a 1/r^2 density, which is possoin
    theta = random.uniform(0, math.pi)
    phi = random.uniform(-math.pi, math.pi)

    #convert to cartesian coordinatges
    x_float = center[0] + r * math.sin(theta) * math.cos(phi)
    y_float = center[1] + r * math.sin(theta) * math.sin(phi)
    z_float = center[2] + r * math.cos(theta)

    # Round to nearest grid index
    ix = int(round(x_float))
    iy = int(round(y_float))
    iz = int(round(z_float))
    
    return ix, iy, iz

def create_possoin_undersampling_mask(r_min, r_max, acceleration_factor, sampled_mask):
    sampled_points = np.sum(sampled_mask)
    
    # compute the center of the original mask
    nx, ny, nz = sampled_mask.shape
    center = np.array([nx // 2, ny // 2, nz // 2])
    
    # Create a grid for distance calculation
    x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    distances = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    
    undersampled_mask = np.zeros(sampled_points.shape) 
    
    # Keep everything inside fixed radius
    undersampled_mask = distances <= r_min
    undersampled_mask = undersampled_mask.astype(int)
    
    #### next I randomly add a point ####
    computed_acceleration_factor = sampled_points/np.sum(undersampled_mask)

    while computed_acceleration_factor > acceleration_factor:
        ix, iy, iz = generate_random_possoin_indices(r_min, r_max, undersampled_mask.shape)
        undersampled_mask[ix, iy, iz] = sampled_mask[ix, iy, iz]
        computed_acceleration_factor = sampled_points/np.sum(undersampled_mask)
    print("Real AF:")
    print(computed_acceleration_factor)
    
    return undersampled_mask

def fully_sampled_mask(fixed_radius,shape):
    '''This function was originally used to create the ground truth data
    which has been measured in spherical k-space.'''
    nx, ny, nz = shape
    center = np.array([nx // 2, ny // 2, nz // 2])
    
    # Create a grid for distance calculation
    x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    distances = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)

    # Fixed sphere: Keep all points inside the sphere
    fixed_sphere_mask = distances <= fixed_radius
    fixed_sphere_mask = fixed_sphere_mask.astype(int)
    
    return fixed_sphere_mask
    
