from scipy.io import loadmat
import numpy as np
from data_undersampling import apply_undersampling
import torch
import scipy.ndimage

##############################################################################
#                         Helper Functions for Rotation                      #
##############################################################################

def random_unit_vector_3d():
    """
    Returns a random unit vector in 3D, uniformly distributed on the unit sphere.
    """
    xyz = np.random.normal(size=3)
    return xyz / np.linalg.norm(xyz)

def rotation_matrix_from_axis_angle(axis, angle_degs):
    """
    Compute the 3x3 rotation matrix for a rotation of `angle_degs` about the 3D 
    unit vector `axis`.
    """
    angle = np.deg2rad(angle_degs)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.array([
        [c + x*x*(1-c),   x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s, c + y*y*(1-c),   y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)]
    ])
    return R

##############################################################################
#                    Affine Transformation Functions                         #
##############################################################################

def affine_transform_3d_complex(vol, M, mode='constant', cval=0.0, order=0):
    """
    Apply a 3D affine transform to a COMPLEX volume `vol` using a 3x3 matrix `M`.
    """
    X, Y, Z = vol.shape[:3]
    center = np.array([X/2.0, Y/2.0, Z/2.0])
    offset = center - M @ center
    real_part = vol.real
    imag_part = vol.imag
    real_rot = scipy.ndimage.affine_transform(
        real_part, matrix=M, offset=offset, order=order, mode=mode, cval=cval)
    imag_rot = scipy.ndimage.affine_transform(
        imag_part, matrix=M, offset=offset, order=order, mode=mode, cval=cval)
    return real_rot + 1j * imag_rot

def affine_transform_3d_mask(vol, M, mode='constant', cval=0.0):
    """
    Apply a 3D affine transform to a real (or binary) volume `vol` using nearest-neighbor interpolation.
    """
    X, Y, Z = vol.shape[:3]
    center = np.array([X/2.0, Y/2.0, Z/2.0])
    offset = center - M @ center
    return scipy.ndimage.affine_transform(
        vol, matrix=M, offset=offset, order=0, mode=mode, cval=cval)

##############################################################################
#                      Helpers for Shifting, Phase, Scaling                  #
##############################################################################

def random_shift_xy(arr, shift_x, shift_y, cval=0):
    """
    Shift `arr` by shift_x along axis 0 (X) and by shift_y along axis 1 (Y), 
    using constant padding with the given cval.
    
    This function works for both real and complex arrays.
    """
    # Create a shift tuple: shift first two dimensions; others unchanged.
    shift_tuple = (shift_x, shift_y) + (0,) * (arr.ndim - 2)
    
    if np.iscomplexobj(arr):
        real_shifted = scipy.ndimage.shift(arr.real, shift=shift_tuple, order=0, mode='constant', cval=cval)
        imag_shifted = scipy.ndimage.shift(arr.imag, shift=shift_tuple, order=0, mode='constant', cval=cval)
        return real_shifted + 1j * imag_shifted
    else:
        return scipy.ndimage.shift(arr, shift=shift_tuple, order=0, mode='constant', cval=cval)

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
#             Main Function: Transform 6D Data, Labels, Mask, and Coordinates  #
##############################################################################

def transform_6d_data_labels_mask(data_6d, labels_6d, mask_6d, rel_x_6d, rel_y_6d):
    """
    Apply identical random transformations to:
      1) data_6d (complex)
      2) labels_6d (complex)
      3) mask_6d (real/binary)
      4) rel_x_6d and rel_y_6d (real coordinate maps)
    
    Each array is expected to have shape (X, Y, Z, t, T, D).
    For each d in [0 .. D-1], we:
      1) Randomly shift in x,y by {-1, 0, 1}. 
         (For data, labels, and mask, use cval=0; for rel_x and rel_y, use cval=-1.)
      2) (Optional) Random 3D rotation (currently commented out).
      3) Apply a global phase and scale magnitude to data and labels.
         Do NOT apply phase or scaling to mask or coordinate maps.
    
    Returns:
        out_data, out_labels, out_mask, out_rel_x, out_rel_y.
    """
    if not (data_6d.shape == labels_6d.shape == mask_6d.shape == rel_x_6d.shape == rel_y_6d.shape):
        raise ValueError("All input arrays must have the same shape.")
    
    out_data   = np.empty_like(data_6d)
    out_labels = np.empty_like(labels_6d)
    out_mask   = np.empty_like(mask_6d)
    out_rel_x  = np.empty_like(rel_x_6d)
    out_rel_y  = np.empty_like(rel_y_6d)
    
    X, Y, Z, t, T, D = data_6d.shape
    
    for d in range(D):
        # Extract d-th volume from each array; shapes: (X, Y, Z, t, T)
        data_slice   = data_6d[..., d].copy()    # complex
        labels_slice = labels_6d[..., d].copy()    # complex
        mask_slice   = mask_6d[..., d].copy()      # real/binary
        rel_x_slice  = rel_x_6d[..., d].copy()     # real
        rel_y_slice  = rel_y_6d[..., d].copy()     # real
        
        # Collapse time dimensions (t, T) into one: shape becomes (X, Y, Z, t*T)
        data_slice   = data_slice.reshape(X, Y, Z, -1)
        labels_slice = labels_slice.reshape(X, Y, Z, -1)
        mask_slice   = mask_slice.reshape(X, Y, Z, -1)
        rel_x_slice  = rel_x_slice.reshape(X, Y, Z, -1)
        rel_y_slice  = rel_y_slice.reshape(X, Y, Z, -1)
        
        # 1) Random shift in x,y by {-1, 0, 1}
        shift_x = np.random.choice([-1, 0, 1])
        shift_y = np.random.choice([-1, 0, 1])
        for time_idx in range(data_slice.shape[-1]):
            data_slice[..., time_idx]   = random_shift_xy(data_slice[..., time_idx], shift_x, shift_y, cval=0)
            labels_slice[..., time_idx] = random_shift_xy(labels_slice[..., time_idx], shift_x, shift_y, cval=0)
            mask_slice[..., time_idx]   = random_shift_xy(mask_slice[..., time_idx], shift_x, shift_y, cval=0)
            rel_x_slice[..., time_idx]  = random_shift_xy(rel_x_slice[..., time_idx], shift_x, shift_y, cval=-1)
            rel_y_slice[..., time_idx]  = random_shift_xy(rel_y_slice[..., time_idx], shift_x, shift_y, cval=-1)
        
        # 2) (Optional) Random 3D rotation (commented out)
        random_axis  = random_unit_vector_3d()
        random_angle = np.random.uniform(-5, 5)
        R = rotation_matrix_from_axis_angle(random_axis, random_angle)
        for time_idx in range(data_slice.shape[-1]):
            data_slice[..., time_idx]   = affine_transform_3d_complex(data_slice[..., time_idx], M=R, mode='constant', cval=0.0, order=0)
            labels_slice[..., time_idx] = affine_transform_3d_complex(labels_slice[..., time_idx], M=R, mode='constant', cval=0.0, order=0)
            mask_slice[..., time_idx]   = affine_transform_3d_mask(mask_slice[..., time_idx], M=R, mode='constant', cval=0.0)
            rel_x_slice[..., time_idx]  = affine_transform_3d_mask(rel_x_slice[..., time_idx], M=R, mode='constant', cval=-1)
            rel_y_slice[..., time_idx]  = affine_transform_3d_mask(rel_y_slice[..., time_idx], M=R, mode='constant', cval=-1)
        
        # 3) Apply global phase to data and labels only.
        phase = np.random.uniform(0, 2 * np.pi)
        for time_idx in range(data_slice.shape[-1]):
            data_slice[..., time_idx]   = data_slice[..., time_idx] * np.exp(1j * phase)
            labels_slice[..., time_idx] = labels_slice[..., time_idx] * np.exp(1j * phase)
        
        # 4) Scale magnitude for data and labels only.
        scale_factor = np.random.uniform(0.95, 1.05)
        for time_idx in range(data_slice.shape[-1]):
            data_slice[..., time_idx]   = data_slice[..., time_idx] * scale_factor
            labels_slice[..., time_idx] = labels_slice[..., time_idx] * scale_factor
        
        # Reshape back to original time dimensions: (X, Y, Z, t, T)
        data_slice   = data_slice.reshape(X, Y, Z, t, T)
        labels_slice = labels_slice.reshape(X, Y, Z, t, T)
        mask_slice   = mask_slice.reshape(X, Y, Z, t, T)
        rel_x_slice  = rel_x_slice.reshape(X, Y, Z, t, T)
        rel_y_slice  = rel_y_slice.reshape(X, Y, Z, t, T)
        
        # Store in output arrays
        out_data[..., d]   = data_slice
        out_labels[..., d] = labels_slice
        out_mask[..., d]   = mask_slice
        out_rel_x[..., d]  = rel_x_slice
        out_rel_y[..., d]  = rel_y_slice
    
    return out_data, out_labels, out_mask, out_rel_x, out_rel_y

##############################################################################
#                  Other Helper Functions: Phase and Scaling                 #
##############################################################################

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





    
    