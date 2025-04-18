from scipy.io import loadmat
import numpy as np
from data_undersampling import apply_undersampling
import torch

##############################################################################
#                               Helper Functions                             #
##############################################################################

import numpy as np
import scipy.ndimage

##############################################################################
#                         Helper Functions for Rotation                      #
##############################################################################

import numpy as np
import scipy.ndimage

##############################################################################
#                           Arbitrary-Axis Rotation                          #
##############################################################################

def random_unit_vector_3d():
    """
    Returns a random unit vector in 3D, uniformly distributed on the unit sphere.
    """
    # Method: sample from normal(0,1), then normalize
    xyz = np.random.normal(size=3)
    return xyz / np.linalg.norm(xyz)

def rotation_matrix_from_axis_angle(axis, angle_degs):
    """
    Compute the 3x3 rotation matrix for a rotation of `angle_degs` about the 3D 
    unit vector `axis`.
    
    - axis: 3-element array, assumed to be a normalized (unit) vector
    - angle_degs: rotation angle in degrees
    """
    angle = np.deg2rad(angle_degs)  # convert to radians
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    # Rodrigues' rotation formula components
    R = np.array([
        [c + x*x*(1-c),     x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s,   c + y*y*(1-c),   y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s,   z*y*(1-c) + x*s, c + z*z*(1-c)]
    ])
    return R

def affine_transform_3d_complex(vol, M, mode='constant', cval=0.0, order=0):
    """
    Apply a 3D affine transform to a COMPLEX volume `vol` using a 3x3 matrix `M`.
    We rotate around the volume center:
    
      1) Shift to make the center of `vol` the origin
      2) Apply the affine transform
      3) Shift back
    
    - vol: array of shape (X, Y, Z), complex
    - M:   3x3 rotation/affine matrix
    - mode, cval, order: passed to `affine_transform` for interpolation
    """
    # Volume shape
    X, Y, Z = vol.shape[:3]
    
    # Compute center of the volume (floating-point)
    center = np.array([X/2.0, Y/2.0, Z/2.0])
    
    # We need a full 3x4 or 4x4 transform for ndimage.affine_transform offset handling.
    # For convenience, we supply the `matrix` and `offset` params in a way that:
    #   new_coords = M * (old_coords - center) + center
    # that means:
    #   new_coords = M*old_coords - M*center + center
    # => offset = center - M*center
    offset = center - M @ center
    
    # Separate real and imaginary
    real_part = vol.real
    imag_part = vol.imag
    
    # Apply affine to each part
    real_rot = scipy.ndimage.affine_transform(
        real_part, 
        matrix=M,
        offset=offset,
        order=order, 
        mode=mode, 
        cval=cval
    )
    imag_rot = scipy.ndimage.affine_transform(
        imag_part, 
        matrix=M,
        offset=offset,
        order=order,
        mode=mode, 
        cval=cval
    )
    
    return real_rot + 1j * imag_rot

def affine_transform_3d_mask(vol, M, mode='constant', cval=0.0):
    """
    Apply a 3D affine transform to a real (or binary) mask `vol`.
    Uses nearest-neighbor interpolation (order=0).
    """
    X, Y, Z = vol.shape[:3]
    center = np.array([X/2.0, Y/2.0, Z/2.0])
    offset = center - M @ center
    
    return scipy.ndimage.affine_transform(
        vol,
        matrix=M,
        offset=offset,
        order=0,   # nearest-neighbor, preserving binary
        mode=mode,
        cval=cval
    )

##############################################################################
#                  Other Helpers: Shift, Phase, Scaling                      #
##############################################################################

def random_shift_xy(arr, shift_x, shift_y):
    """
    Shift `arr` by shift_x along axis 0 (X) and by shift_y along axis 1 (Y), using constant
    padding (0) for out-of-bound values. This function works for both real and complex arrays.
    """
    # Create a shift tuple that only shifts the first two dimensions, leaving others unchanged.
    shift_tuple = (shift_x, shift_y) + (0,) * (arr.ndim - 2)
    
    if np.iscomplexobj(arr):
        # Shift the real and imaginary parts separately
        real_shifted = scipy.ndimage.shift(arr.real, shift=shift_tuple, order=0, mode='constant', cval=0)
        imag_shifted = scipy.ndimage.shift(arr.imag, shift=shift_tuple, order=0, mode='constant', cval=0)
        return real_shifted + 1j * imag_shifted
    else:
        return scipy.ndimage.shift(arr, shift=shift_tuple, order=0, mode='constant', cval=0)


def random_global_phase(arr, phase_radians):
    """
    Multiply `arr` (complex) by exp(i*phase_radians).
    """
    return arr * np.exp(1j * phase_radians)

def random_scale_magnitude(arr, scale_factor):
    """
    Scale the magnitude of `arr` (complex) by `scale_factor` (real).
    This does not affect the phase.
    """
    return arr * scale_factor

##############################################################################
#              Main Function (One Random 3D Rotation per slice)              #
##############################################################################

def transform_6d_data_labels_mask(data_6d, labels_6d, mask_6d):
    """
    Apply identical random transformations to:
      1) `data_6d` (complex)
      2) `labels_6d` (complex)
      3) `mask_6d` (real or binary)
    
    Each has shape (X, Y, Z, t, T, D).
    
    For each d in [0..D-1], we:
      1) Randomly shift in x,y by {-1, 0, 1}.
      2) Randomly pick a random axis (unit vector) in 3D,
         pick a random angle in [-10..10] degrees,
         apply exactly ONE rotation about that axis.
      3) Randomly pick a global phase [0..2π] => multiply data & labels by exp(i * phase).
      4) Randomly pick a scale factor in [0.95..1.05] => scale magnitude of data & labels.
    
    The mask is shifted & rotated with the same parameters, but
    we do NOT apply phase or scale to the mask.
    
    Returns (out_data, out_labels, out_mask).
    """
    if not (data_6d.shape == labels_6d.shape == mask_6d.shape):
        raise ValueError("All three arrays must have the same shape.")
    
    out_data   = np.empty_like(data_6d)
    out_labels = np.empty_like(labels_6d)
    out_mask   = np.empty_like(mask_6d)
    
    X, Y, Z, t, T, D = data_6d.shape
    
    for d in range(D):
        # Extract each 3D+time slice
        data_slice   = data_6d[..., d].copy()   # shape (X, Y, Z, t, T), complex
        labels_slice = labels_6d[..., d].copy() # shape (X, Y, Z, t, T), complex
        mask_slice   = mask_6d[..., d].copy()   # shape (X, Y, Z, t, T), real/binary
        
        # For convenience, we can flatten the time dims and treat the first 3 as spatial:
        #   new_shape = (X, Y, Z) + (t*T,) 
        # We'll rotate each "voxel time-series" in the same way in 3D space.
        # Or if you truly want to rotate each time sample separately, you'd loop over t,T.
        # 
        # Here, I'm just going to reshape to (X, Y, Z) *for each time point in [t*T].
        # Then apply the same rotation for each time slice. 
        # This ensures that the 3D rotation is consistent across the time dimension.
        
        data_slice   = data_slice.reshape(X, Y, Z, -1)     # shape: (X, Y, Z, t*T)
        labels_slice = labels_slice.reshape(X, Y, Z, -1)   
        mask_slice   = mask_slice.reshape(X, Y, Z, -1)
        
        # 1) SHIFT in x,y by a small random integer
        shift_x = np.random.choice([-1, 0, 1])
        shift_y = np.random.choice([-1, 0, 1])
        
        for time_idx in range(data_slice.shape[-1]):
            data_slice[..., time_idx]   = random_shift_xy(data_slice[..., time_idx],   shift_x, shift_y)
            labels_slice[..., time_idx] = random_shift_xy(labels_slice[..., time_idx], shift_x, shift_y)
            mask_slice[..., time_idx]   = random_shift_xy(mask_slice[..., time_idx],   shift_x, shift_y)
        
#         # 2) SINGLE RANDOM 3D ROTATION around an arbitrary axis
        random_axis  = random_unit_vector_3d()              # pick a random unit vector
        random_angle = np.random.uniform(-5, 5)           # degrees
        
        # Build rotation matrix
        R = rotation_matrix_from_axis_angle(random_axis, random_angle)
        
        # Apply once for each time index
        for time_idx in range(data_slice.shape[-1]):
            # data & labels: complex
            data_slice[..., time_idx]   = affine_transform_3d_complex(
                data_slice[..., time_idx],
                M=R,
                mode='constant',
                cval=0.0,
                order=0  # linear interpolation for complex magnitude
            )
            labels_slice[..., time_idx] = affine_transform_3d_complex(
                labels_slice[..., time_idx],
                M=R,
                mode='constant',
                cval=0.0,
                order=0
            )
            # mask: real/binary
            mask_slice[..., time_idx] = affine_transform_3d_mask(
                mask_slice[..., time_idx],
                M=R,
                mode='constant',
                cval=0.0
            )
        
        # 3) GLOBAL PHASE (data & labels only, not mask)
        
        phase = np.random.uniform(0, 2 * np.pi)
        for time_idx in range(data_slice.shape[-1]):
            data_slice[..., time_idx]   = random_global_phase(data_slice[..., time_idx],   phase)
            labels_slice[..., time_idx] = random_global_phase(labels_slice[..., time_idx], phase)
        
        # 4) SCALING MAGNITUDE (data & labels only, not mask)
        
        scale_factor = np.random.uniform(0.095, 1.05)
        for time_idx in range(data_slice.shape[-1]):
            data_slice[..., time_idx]   = random_scale_magnitude(data_slice[..., time_idx],   scale_factor)
            labels_slice[..., time_idx] = random_scale_magnitude(labels_slice[..., time_idx], scale_factor)
        
        # Reshape back to (X, Y, Z, t, T)
        data_slice   = data_slice.reshape(X, Y, Z, t, T)
        labels_slice = labels_slice.reshape(X, Y, Z, t, T)
        mask_slice   = mask_slice.reshape(X, Y, Z, t, T)
        
        # Store in output arrays
        out_data[..., d]   = data_slice
        out_labels[..., d] = labels_slice
        out_mask[..., d]   = mask_slice
    
    return out_data, out_labels, out_mask






    
    