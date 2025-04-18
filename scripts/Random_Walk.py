from scipy.io import loadmat
import numpy as np
from data_undersampling import apply_undersampling
import torch

#### Here I implement the entire random walk strategy for generating more training data in the zfT case ####

def generate_direction():
    directions = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if (dx, dy, dz) == (0, 0, 0):
                    continue  # skip zero vector
                directions.append((dx, dy, dz))
    return directions  # length = 26

def extract_line_data(volume, coords):
    """
    volume: np.ndarray with shape (X, Y, Z, f, T, C)
    coords: list of (x, y, z)
    returns shape: (line_length, f, T, C)
    """
    return np.stack([volume[x, y, z, :, :, :] for (x, y, z) in coords], axis=0)

def build_step_directions():
    directions = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if (dx, dy, dz) == (0, 0, 0):
                    continue  # skip zero vector
                directions.append((dx, dy, dz))
    return directions  # length = 26
    
def random_walk_line_3d(X, Y, Z, line_length=22, max_tries=10):
    """
    Perform a random walk in a 3D grid of shape (X,Y,Z) for `line_length` steps.
    Returns:
        coords : list of (x,y,z) of length `line_length`, or None if failed after max_tries.
    """
    directions = build_step_directions()  # 26 directions
    for attempt in range(max_tries):
        # 1) Random start
        x = np.random.randint(0, X)
        y = np.random.randint(0, Y)
        z = np.random.randint(0, Z)

        coords = [(x, y, z)]

        # 2) Pick an initial direction at random
        d = directions[np.random.randint(len(directions))]

        for step in range(line_length - 1):
            # Next possible directions = all directions except the immediate opposite of d
            # The opposite direction to d is (-d_x, -d_y, -d_z)
            d_opposite = (-d[0], -d[1], -d[2])
            valid_dirs = [dir_ for dir_ in directions if dir_ != d_opposite]

            # Pick a random direction from valid_dirs
            d = valid_dirs[np.random.randint(len(valid_dirs))]

            # Move
            x += d[0]
            y += d[1]
            z += d[2]

            # Check bounds
            if x < 0 or x >= X or y < 0 or y >= Y or z < 0 or z >= Z:
                break  # out of bounds; break from this line
            coords.append((x, y, z))

        # Check if we got a full line
        if len(coords) == line_length:
            return coords

    return None  # if all attempts failed to produce a line of full length    

def generate_fixed_random_lines(data, labels, masks, num_trajectories=50, line_length=21):
    """
    Generates a fixed number of random walk trajectories per subject.

    Parameters
    ----------
    data : np.ndarray
        Shape (X, Y, Z, f, T, C, N) -> original spatial + spectral data with N subjects.
    labels : np.ndarray
        Same shape as data, contains ground truth.
    masks : np.ndarray
        Same shape as data, contains masks.
    num_trajectories : int
        Number of random trajectories per subject per epoch.
    line_length : int
        The fixed length of each extracted trajectory.

    Returns
    -------
    line_data, line_labels, line_masks : np.ndarray
        All with shape (N * num_trajectories, C, line_length, f, T).
    """
    # Get dimensions; note that N is the last dimension.
    X, Y, Z, f, T, C, N = data.shape

    line_data_list = []
    line_label_list = []
    line_mask_list = []

    # Loop over subjects
    for n in range(N):
        subject_data = data[..., n]    # shape: (X, Y, Z, f, T, C)
        subject_labels = labels[..., n]
        subject_masks = masks[..., n]

        for _ in range(num_trajectories):
            # Generate one random line for this subject
            coords = random_walk_line_3d(X, Y, Z, line_length=line_length, max_tries=10)
            if coords is None:
                continue  # Skip if no valid line was found

            # Extract data for this line; each has shape (line_length, f, T, C)
            line_data = extract_line_data(subject_data, coords)
            line_label = extract_line_data(subject_labels, coords)
            line_mask = extract_line_data(subject_masks, coords)

            # Reorder axes to (C, line_length, f, T)
            line_data = np.transpose(line_data, (3, 0, 1, 2))
            line_label = np.transpose(line_label, (3, 0, 1, 2))
            line_mask = np.transpose(line_mask, (3, 0, 1, 2))

            line_data_list.append(line_data)
            line_label_list.append(line_label)
            line_mask_list.append(line_mask)

    # Stack all generated lines from all subjects
    line_data = np.stack(line_data_list, axis=0)   # shape: (total_samples, C, line_length, f, T)
    line_label = np.stack(line_label_list, axis=0)
    line_mask = np.stack(line_mask_list, axis=0)

    return line_data, line_label, line_mask