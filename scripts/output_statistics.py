from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from data_preparation import *
from tensorboard.backend.event_processing import event_accumulator
from skimage.metrics import structural_similarity as ssim 

#### data statistics ####

def MSE_time_domain(model_pred, ground_truth, norm_values, average_over_T = False, normalize = False):
    """
    Compute MSE for model predictions (has to be in format x,y,z,t,T).

    Parameters:
        model_pred: Model predictions on test set
        ground_truth: ground truth
        norm_values: normalization values
        average_over_T: If True: Compute the average MSE for different t values over T.
                        If False: Compute overall MSE
        normalize: Normalize 3D Volumes by dividing by the largest absolute value of the ground truth
                    Apply same normalization to the model output - then compute MSE.

    Returns:
        MSE, or averaged MSE over T.
    """
    x, y, z, t, T = ground_truth.shape
    
    if normalize:
        max_abs_value_ground_truth =  norm_values #np.max(np.abs(NN_input_test), axis=(0, 1, 2), keepdims=True)
        ground_truth = ground_truth / max_abs_value_ground_truth
        model_pred = model_pred / max_abs_value_ground_truth
    
    error = np.abs(model_pred - ground_truth)**2
    mse_per_spatial_volume = np.mean(error, axis=(0, 1, 2))
    
    if average_over_T == False:
        final_mse = np.sum(mse_per_spatial_volume)/(2*T*t) #divison by 2 two make it comparable to network output, which has complex and real channel
        return final_mse
    
    else:
        mse_vector = np.mean(mse_per_spatial_volume, axis=1) # averages over T
        return mse_vector
    
def MSE_spectral_domain(model_pred, ground_truth, norm_values, average_over_T = False, normalize = False):
    """
    Compute MSE for model predictions in spectral domain (has to be in format x,y,z,t,T).
    The model_pred is automatically 0 filled along t to match ground_truth in case they have
    divergent shape.

    Parameters:
        model_pred: Model predictions on test set
        ground_truth: ground truth - dont use trancuated t version for spectral domain
        norm_values: normalization values
        average_over_T: If True: Compute the average MSE for different t values over T.
                        If False: Compute overall MSE
        normalize: Normalize in the spectral domain (divide by maximum aboslute value in spectral domain)

    Returns:
        MSE in spectral domain, or averaged MSE over T.
    """
    t_ground_truth = ground_truth.shape[3]
    t_model =  model_pred.shape[3]
    
    if t_model < t_ground_truth: # apply zero filling
        extended_output = np.zeros(ground_truth.shape, dtype=model_pred.dtype)
        extended_output[..., :t_model, :] =  model_pred
        model_pred = extended_output
        
    ground_truth_spectral = np.fft.fftshift(np.fft.fft(ground_truth, axis=-2), axes=-2)
    model_pred_spectral = np.fft.fftshift(np.fft.fft(model_pred, axis=-2), axes=-2)
    
    MSE = MSE_time_domain(model_pred_spectral, ground_truth_spectral, norm_values, average_over_T = average_over_T, normalize = normalize) # Note that this can also be applied in the spectral domain, avoiding duplicated code
    
    return MSE

def plot_general_statistics(model_pred, model_input, full_ground_truth, truncate_t, norm_values, label = "Model Output", label2 = "IFFT"):
    """
    This function creates a single row of 3 plots with two series each:
        1. MSE in time domain (non-normalized)
        2. MSE in time domain (normalized)
        3. MSE in spectral domain (without normalization)
    
    For each plot:
       - One curve for 'model_pred' (network output)
       - One curve for 'model_input' (network input)
    
    Parameters:
        model_pred:    Model predictions on test set
        model_input:   Network input (undersampled data or similar) for comparison
        full_ground_truth: Ground truth (don't use truncated t version for spectral domain)
        truncate_t:    How many time steps t were used in the model training (and thus in the prediction)
        norm_values:   Normalization values for time domain
        label:         Change label of the model prediction
        label2:        Change label of the IFFT (model input)

    Returns:
        A single row of 3 plots comparing MSE of 'model_pred' vs. 'model_input'
    """

    # 1) Truncate the ground truth to match shape (only for time-domain comparisons)
    ground_truth = full_ground_truth[:, :, :, :truncate_t, :]

    # 2) Compute MSE for 'model_pred' in time domain
    mse_time_non_normalized_pred = MSE_time_domain(
        model_pred, ground_truth, norm_values, average_over_T=True, normalize=False
    )
    mse_time_normalized_pred = MSE_time_domain(
        model_pred, ground_truth, norm_values, average_over_T=True, normalize=True
    )
    
    #   - Totals across all T
    mse_time_non_normalized_total_pred = MSE_time_domain(
        model_pred, ground_truth, norm_values, average_over_T=False, normalize=False
    )
    mse_time_normalized_total_pred = MSE_time_domain(
        model_pred, ground_truth, norm_values, average_over_T=False, normalize=True
    )

    # 3) Compute MSE for 'model_pred' in spectral domain
    #    Use full_ground_truth (un-truncated) so we have the entire spectral axis
    mse_spectral_pred = MSE_spectral_domain(
        model_pred, full_ground_truth, norm_values, average_over_T=True, normalize=False
    )
    mse_spectral_total_pred = MSE_spectral_domain(
        model_pred, full_ground_truth, norm_values, average_over_T=False, normalize=False
    )

    # 4) Compute MSE for 'model_input' in time domain
    mse_time_non_normalized_inp = MSE_time_domain(
        model_input, ground_truth, norm_values, average_over_T=True, normalize=False
    )
    mse_time_normalized_inp = MSE_time_domain(
        model_input, ground_truth, norm_values, average_over_T=True, normalize=True
    )
    
    #   - Totals across all T
    mse_time_non_normalized_total_inp = MSE_time_domain(
        model_input, ground_truth, norm_values, average_over_T=False, normalize=False
    )
    mse_time_normalized_total_inp = MSE_time_domain(
        model_input, ground_truth, norm_values, average_over_T=False, normalize=True
    )

    # 5) Compute MSE for 'model_input' in spectral domain
    mse_spectral_inp = MSE_spectral_domain(
        model_input, full_ground_truth, norm_values, average_over_T=True, normalize=False
    )
    mse_spectral_total_inp = MSE_spectral_domain(
        model_input, full_ground_truth, norm_values, average_over_T=False, normalize=False
    )

    # 6) Prepare the figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("MSE in Time and Spectral Domains", fontsize=16)

    # Plot 1: MSE in time domain (non-normalized)
    axes[0].plot(mse_time_non_normalized_pred, marker='o', linestyle='-', label=label)
    axes[0].plot(mse_time_non_normalized_inp,  marker='x', linestyle='--', label=label2)
    axes[0].set_title("Time Domain (Non-Normalized)")
    axes[0].set_xlabel("t Index")
    axes[0].set_ylabel("MSE")
    axes[0].grid(True)
    axes[0].legend()

    # Plot 2: MSE in time domain (normalized)
#     axes[1].plot(mse_time_normalized_pred, marker='o', linestyle='-', label=label)
#     axes[1].plot(mse_time_normalized_inp,  marker='x', linestyle='--', label=label2)
#     axes[1].set_title("Time Domain (Normalized)")
#     axes[1].set_xlabel("t Index")
#     axes[1].set_ylabel("MSE")
#     axes[1].grid(True)
#     axes[1].legend()

    # Plot 3: MSE in spectral domain (no normalization)
    axes[1].plot(mse_spectral_pred, marker='o', linestyle='-', label=label)
    axes[1].plot(mse_spectral_inp,  marker='x', linestyle='--', label=label2)
    axes[1].set_title("Spectral Domain (No Normalization)")
    axes[1].set_xlabel("Spectral Index")
    axes[1].set_ylabel("MSE")
    axes[1].grid(True)
    axes[1].legend()

    # 7) Adjust layout and show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # 8) Print total MSE values in the console
    print("==== Model Output vs Ground Truth ====")
    print(f"Total MSE in image domain: {mse_time_non_normalized_total_pred}")
    print(f"Normalized Total MSE in image domain: {mse_time_normalized_total_pred}")
    print(f"Total MSE in spectral domain: {mse_spectral_total_pred}\n")

    print("==== Model Input vs Ground Truth ====")
    print(f"Total MSE in image domain: {mse_time_non_normalized_total_inp}")
    print(f"Normalized Total MSE in image domain: {mse_time_normalized_total_inp}")
    print(f"Total MSE in spectral domain: {mse_spectral_total_inp}")
    
def comparison_Plot_3D(model_pred, ground_truth, tf, T, domain = "time"):  
    """
    This function creates a two colum plot showing a comparison between the model output
    and the groundtruth

    Parameters:
        model_pred: Model predictions on test set
        ground_truth: ground truth
        tf: specify either the t index (time domain), or f index (spectral domain)
        T: specify metabolism time index
        domain: specify if you want to compare in the spectral or time domain.
                if spectral is chosen, the arrays are automatically transformed

    Returns:
        Plot as described.
    """
    if domain not in["time","spectral"]:
        print("Only domain time or spectral is allowed")
        
    if domain == "time":
        pass
    
    else:
        if  model_pred.shape[3] < ground_truth.shape[3]: # apply zero filling
            extended_output = np.zeros(ground_truth.shape, dtype=model_pred.dtype)
            extended_output[..., :model_pred.shape[3], :] =  model_pred
            model_pred = extended_output
            ## now transform to spectral domain
        model_pred = np.fft.fftshift(np.fft.fft(model_pred, axis=-2), axes=-2)
        ground_truth = np.fft.fftshift(np.fft.fft(ground_truth, axis=-2), axes=-2)
             
    abs_output = np.abs(model_pred[:,:,:, tf,T])  # Absolute value of output
    abs_labels = np.abs(ground_truth[:,:,:, tf,T])  # Absolute value of labels
    
    # Parameters
    z_indices = range(abs_labels.shape[2])  # Number of slices along the z-dimension
    n_cols = 2  # Two columns: ground truth and output
    n_rows = len(z_indices)  # One row per z index

    # Create a figure for visualization
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3 * n_rows))
    axes = axes.reshape(n_rows, n_cols)  # Reshape axes to 2D array for easier indexing

    # Loop through slices along the last dimension (z-dimension)
    for i, z in enumerate(z_indices):
        # Extract 2D slices for the current z index
        slice_gt = abs_labels[:, :, z]  # Ground truth slice
        slice_output = abs_output[:, :, z]  # Output slice

        # Plot ground truth slice
        ax_gt = axes[i, 0]
        im_gt = ax_gt.imshow(slice_gt, cmap='viridis', origin='lower')
        ax_gt.set_title(f"Ground Truth (z={z})")
        ax_gt.axis("off")
        fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)

        # Plot model output slice
        ax_output = axes[i, 1]
        im_output = ax_output.imshow(slice_output, cmap='viridis', origin='lower')
        ax_output.set_title(f"Model Output (z={z})")
        ax_output.axis("off")
        fig.colorbar(im_output, ax=ax_output, fraction=0.046, pad=0.04)

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()
    
def comparison_Plot_3D_vs_Ifft(model_pred, ground_truth, model_input, tf, T, domain="time", 
                               label="Model Output", label2="IFFT", label3="Ground Truth", fixed_scale = False):
    """
    This function creates a three-column plot showing a comparison between the model output,
    the ground truth, and the inverse FFT (IFFT) input, along with PSNR, SSIM, and NRMSE metrics.
    The color scales for the model output and IFFT plots are determined by the corresponding 
    ground-truth slice’s min and max values.
    if fixed_scale = True: Determine colorbar by ground truth
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    # Define RMSE and NRMSE calculation functions
    def calculate_rmse(pred, gt):
        return np.sqrt(np.mean((pred - gt) ** 2))

    def calculate_nrmse(pred, gt):
        return calculate_rmse(pred, gt) / (gt.max() - gt.min())

    # Ensure domain is valid
    if domain not in ["time", "spectral"]:
        print("Only 'time' or 'spectral' domain is allowed.")
        return

    # If 'spectral' domain is chosen, transform the data accordingly
    if domain == "spectral":
        # Zero-filling if necessary
        if model_pred.shape[3] < ground_truth.shape[3]:
            extended_output = np.zeros(ground_truth.shape, dtype=model_pred.dtype)
            extended_output[..., :model_pred.shape[3], :] = model_pred
            model_pred = extended_output

        # FFT along the specified axis and shift
        model_pred = np.fft.fftshift(np.fft.fft(model_pred, axis=-2), axes=-2)
        ground_truth = np.fft.fftshift(np.fft.fft(ground_truth, axis=-2), axes=-2)
        model_input = np.fft.fftshift(np.fft.fft(model_input, axis=-2), axes=-2)

    # Take absolute values for visualization and metric computation
    abs_output = np.abs(model_pred[:, :, :, tf, T])
    abs_labels = np.abs(ground_truth[:, :, :, tf, T])
    abs_input = np.abs(model_input[:, :, :, tf, T])

    # Slices along z-axis (the 3rd dimension here)
    z_indices = range(abs_labels.shape[2])  # z-dim

    # We will have 3 columns: Ground Truth, Model Output, Model Input (IFFT)
    n_cols = 3
    n_rows = len(z_indices)

    # Create figure with adjustable size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    # In case n_rows=1, axes might be 1D, so force it to be 2D
    if n_rows == 1:
        axes = np.array([axes])

    for i, z in enumerate(z_indices):
        # Extract 2D slice at index z
        slice_gt = abs_labels[:, :, z]
        slice_output = abs_output[:, :, z]
        slice_input = abs_input[:, :, z]

        # Define normalization based on ground-truth slice
        slice_min, slice_max = slice_gt.min(), slice_gt.max()
        # Avoid degenerate range
        if slice_min == slice_max:
            slice_min, slice_max = 0, 1e-10

        # --- Ground Truth ---
        ax_gt = axes[i, 0]
        if fixed_scale:
            im_gt = ax_gt.imshow(slice_gt, cmap='viridis', origin='lower', 
                                 vmin=slice_min, vmax=slice_max)
        else:
            im_gt = ax_gt.imshow(slice_gt, cmap='viridis', origin='lower')
        ax_gt.set_title(f"{label3} (z={z})")
        #ax_gt.axis("off")
        ax_gt.grid(True)
        fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)

        # Compute PSNR, SSIM, and NRMSE for Model Output
        data_range = slice_max - slice_min
        psnr_out = peak_signal_noise_ratio(slice_gt, slice_output, data_range=data_range)
        ssim_out = structural_similarity(slice_gt, slice_output, data_range=data_range)
        nrmse_out = calculate_nrmse(slice_output, slice_gt)

        # --- Model Output ---
        ax_output = axes[i, 1]
        if fixed_scale:
            im_output = ax_output.imshow(slice_output, cmap='viridis', origin='lower', 
                                         vmin=slice_min, vmax=slice_max)
        else:
            im_output = ax_output.imshow(slice_output, cmap='viridis', origin='lower')
        ax_output.set_title(
            f"{label} (z={z})\nPSNR={psnr_out:.2f}, SSIM={ssim_out:.3f}, NRMSE={nrmse_out:.3f}"
        )
        #ax_output.axis("off")
        ax_output.grid(True)
        fig.colorbar(im_output, ax=ax_output, fraction=0.046, pad=0.04)

        # Compute PSNR, SSIM, and NRMSE for IFFT Input
        psnr_in = peak_signal_noise_ratio(slice_gt, slice_input, data_range=data_range)
        ssim_in = structural_similarity(slice_gt, slice_input, data_range=data_range)
        nrmse_in = calculate_nrmse(slice_input, slice_gt)

        # --- Model Input (IFFT) ---
        ax_input = axes[i, 2]
        if fixed_scale:
            im_input = ax_input.imshow(slice_input, cmap='viridis', origin='lower', 
                                       vmin=slice_min, vmax=slice_max)
        else:
            im_input = ax_input.imshow(slice_input, cmap='viridis', origin='lower')
        ax_input.set_title(
            f"{label2} (z={z})\nPSNR={psnr_in:.2f}, SSIM={ssim_in:.3f}, NRMSE={nrmse_in:.3f}"
        )
        #ax_input.axis("off")
        ax_input.grid(True)
        fig.colorbar(im_input, ax=ax_input, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

def REScomparison_Plot_3D_vs_Ifft(
    model_pred,
    ground_truth,
    model_input,
    tf,
    T,
    domain="time",
    label="Model Output",
    label2="IFFT",
    label3="Ground Truth",
):
    """
    This function creates a three-column plot:
      1) The ground-truth slice,
      2) The residual between the model output and ground truth,
      3) The residual between the IFFT (model_input) and ground truth.
    It also computes PSNR, SSIM, and NRMSE metrics (still between the 
    model output/IFFT input and the ground truth, *not* the residuals). 
    Both residual maps share the same color bar range.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    # Define RMSE and NRMSE calculation functions
    def calculate_rmse(pred, gt):
        return np.sqrt(np.mean((pred - gt) ** 2))

    def calculate_nrmse(pred, gt):
        return calculate_rmse(pred, gt) / (gt.max() - gt.min() + 1e-10)

    # Ensure domain is valid
    if domain not in ["time", "spectral"]:
        print("Only 'time' or 'spectral' domain is allowed.")
        return

    # If 'spectral' domain is chosen, transform the data accordingly
    if domain == "spectral":
        # Zero-filling if necessary
        if model_pred.shape[3] < ground_truth.shape[3]:
            extended_output = np.zeros(ground_truth.shape, dtype=model_pred.dtype)
            extended_output[..., : model_pred.shape[3], :] = model_pred
            model_pred = extended_output

        # FFT along the specified axis and shift
        model_pred = np.fft.fftshift(np.fft.fft(model_pred, axis=-2), axes=-2)
        ground_truth = np.fft.fftshift(np.fft.fft(ground_truth, axis=-2), axes=-2)
        model_input = np.fft.fftshift(np.fft.fft(model_input, axis=-2), axes=-2)

    # Take absolute values for metric computation
    # (You can keep these complex if you want to compare phases, etc.)
    abs_output = np.abs(model_pred[:, :, :, tf, T])
    abs_labels = np.abs(ground_truth[:, :, :, tf, T])
    abs_input = np.abs(model_input[:, :, :, tf, T])

    # Slices along z-axis (the 3rd dimension here)
    z_indices = range(abs_labels.shape[2])  # z-dim

    # We will have 3 columns: Ground Truth, Residual (Model Output), Residual (IFFT)
    n_cols = 3
    n_rows = len(z_indices)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    for i, z in enumerate(z_indices):
        # --- Ground Truth slice ---
        slice_gt = abs_labels[:, :, z]

        # --- Slices for comparison ---
        slice_output = abs_output[:, :, z]
        slice_input = abs_input[:, :, z]

        # --- Compute metrics (model output vs. GT) ---
        data_range = slice_gt.max() - slice_gt.min() + 1e-10
        psnr_out = peak_signal_noise_ratio(slice_gt, slice_output, data_range=data_range)
        ssim_out = structural_similarity(slice_gt, slice_output, data_range=data_range)
        nrmse_out = calculate_nrmse(slice_output, slice_gt)

        # --- Compute metrics (IFFT vs. GT) ---
        psnr_in = peak_signal_noise_ratio(slice_gt, slice_input, data_range=data_range)
        ssim_in = structural_similarity(slice_gt, slice_input, data_range=data_range)
        nrmse_in = calculate_nrmse(slice_input, slice_gt)

        # --- Plot Ground Truth ---
        ax_gt = axes[i, 0]
        im_gt = ax_gt.imshow(
            slice_gt, cmap="viridis", origin="lower"
        )
        ax_gt.set_title(f"{label3} (z={z})")
        ax_gt.axis("off")
        fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)

        # --- Compute residuals (output - GT), (input - GT) ---
        residual_output = np.abs(slice_output - slice_gt)
        residual_input = np.abs(slice_input - slice_gt)

        # Ensure both residual plots share the same color scale
        min_val = min(residual_output.min(), residual_input.min())
        max_val = max(residual_output.max(), residual_input.max())
        # (Optionally use a symmetric colorbar around zero:)
        # max_abs = max(abs(min_val), abs(max_val))
        # min_val, max_val = -max_abs, max_abs

        # --- Plot Residual (Model Output) ---
        ax_res_out = axes[i, 1]
        im_out = ax_res_out.imshow(
            residual_output, cmap="bwr", origin="lower", vmin=min_val, vmax=max_val
        )
        ax_res_out.set_title(
            f"Residual: {label}\n"
            f"(z={z})\nPSNR={psnr_out:.2f}, SSIM={ssim_out:.3f}, NRMSE={nrmse_out:.3f}"
        )
        ax_res_out.axis("off")
        fig.colorbar(im_out, ax=ax_res_out, fraction=0.046, pad=0.04)

        # --- Plot Residual (IFFT) ---
        ax_res_in = axes[i, 2]
        im_in = ax_res_in.imshow(
            residual_input, cmap="bwr", origin="lower", vmin=min_val, vmax=max_val
        )
        ax_res_in.set_title(
            f"Residual: {label2}\n"
            f"(z={z})\nPSNR={psnr_in:.2f}, SSIM={ssim_in:.3f}, NRMSE={nrmse_in:.3f}"
        )
        ax_res_in.axis("off")
        fig.colorbar(im_in, ax=ax_res_in, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    
def compare_models(logdirs, model_labels=None, metrics=None, figsize=(10, 5), train=True, test=True):
    """
    Compare multiple models (logdirs) on the MSE metric by plotting train, validation,
    and test curves on a single plot.
    
    Parameters
    ----------
    logdirs : list of str
        List of paths to the log directories (each containing a single tfevents file).
    model_labels : list of str, optional
        Labels to identify each model in the legend; must be same length as logdirs.
        If None, generic labels (e.g., Model 1, Model 2, ...) are used.
    metrics : list of tuple (str, str)
        A list of (plot_title, tensorboard_tag) pairs, e.g.:
            [
                ("Loss Train",      "Loss/Train"),
                ("Loss Validation", "Loss/Validation"),
                ("Loss Test",       "Loss/Test")
            ]
    figsize : tuple
        Figure size, e.g., (width, height).
    train : bool
        Whether to plot the train curve (dashed).
    test : bool
        Whether to plot the test curve (solid).
    """
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing import event_accumulator

    # If labels are not provided, generate generic ones.
    if model_labels is None:
        model_labels = [f"Model {i+1}" for i in range(len(logdirs))]

    # 1) Group metrics by their base name and type.
    grouped_metrics = {}
    for (plot_title, tb_tag) in metrics:
        if tb_tag.endswith("/Train"):
            base = tb_tag[:-len("/Train")]
            if base not in grouped_metrics:
                grouped_metrics[base] = {}
            grouped_metrics[base]["train_title"] = plot_title
            grouped_metrics[base]["train_tag"] = tb_tag

        elif tb_tag.endswith("/Validation"):
            base = tb_tag[:-len("/Validation")]
            if base not in grouped_metrics:
                grouped_metrics[base] = {}
            grouped_metrics[base]["val_title"] = plot_title
            grouped_metrics[base]["val_tag"] = tb_tag

        elif tb_tag.endswith("/Val"):
            base = tb_tag[:-len("/Val")]
            if base not in grouped_metrics:
                grouped_metrics[base] = {}
            grouped_metrics[base]["val_title"] = plot_title
            grouped_metrics[base]["val_tag"] = tb_tag

        elif tb_tag.endswith("/Test"):
            base = tb_tag[:-len("/Test")]
            if base not in grouped_metrics:
                grouped_metrics[base] = {}
            grouped_metrics[base]["test_title"] = plot_title
            grouped_metrics[base]["test_tag"] = tb_tag

        else:
            print(f"Warning: Tag '{tb_tag}' doesn't end in /Train, /Val, /Validation, or /Test. Ignoring.")
            continue

    # 2) Load event accumulators for each log directory.
    accumulators = []
    for logdir in logdirs:
        ea = event_accumulator.EventAccumulator(logdir)
        ea.Reload()
        accumulators.append(ea)

    # 3) Prepare a single figure and axis.
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title("MSE Comparison")
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE")
    #ax.set_yscale("log")  # Using logarithmic scale; change if needed.

    # 4) Create a color cycle for the models.
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    model_colors = [color_cycle[i % len(color_cycle)] for i in range(len(logdirs))]

    # For simplicity, assume we're comparing a single base metric (e.g., "Loss")
    base = list(grouped_metrics.keys())[0] if grouped_metrics else None
    if base is None:
        print("No valid metrics provided.")
        return

    # 5) For each model, plot available curves.
    for model_idx, (label, ea) in enumerate(zip(model_labels, accumulators)):
        color = model_colors[model_idx]

        # Plot train curve (dashed)
        if train and "train_tag" in grouped_metrics[base]:
            tag = grouped_metrics[base]["train_tag"]
            if tag in ea.Tags()["scalars"]:
                data = ea.Scalars(tag)
                steps = [x.step for x in data]
                values = [x.value for x in data]
                ax.plot(steps, values, label=f"{label} (Train)", linestyle="--", color=color)
            else:
                print(f"Warning: Tag '{grouped_metrics[base]['train_tag']}' not found for model '{label}'. Skipping Train.")

        # Plot validation curve (dotted)
        if "val_tag" in grouped_metrics[base]:
            tag = grouped_metrics[base]["val_tag"]
            if tag in ea.Tags()["scalars"]:
                data = ea.Scalars(tag)
                steps = [x.step for x in data]
                values = [x.value for x in data]
                ax.plot(steps, values, label=f"{label} (Validation)", linestyle=":", color=color)
            else:
                print(f"Warning: Tag '{grouped_metrics[base]['val_tag']}' not found for model '{label}'. Skipping Validation.")

        # Plot test curve (solid)
        if test and "test_tag" in grouped_metrics[base]:
            tag = grouped_metrics[base]["test_tag"]
            if tag in ea.Tags()["scalars"]:
                data = ea.Scalars(tag)
                steps = [x.step for x in data]
                values = [x.value for x in data]
                ax.plot(steps, values, label=f"{label} (Test)", linestyle="-", color=color)
            else:
                print(f"Warning: Tag '{grouped_metrics[base]['test_tag']}' not found for model '{label}'. Skipping Test.")

    # Add legend if any handles exist.
    handles, labels_handles = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels_handles)
    else:
        print("No handles with labels found to put in legend.")

    plt.tight_layout()
    plt.show()





def PSNR_time_domain(model_pred, ground_truth, norm_values, average_over_T = False, normalize = False):
    """
    Compute PSNR for model predictions (has to be in format x,y,z,t,T).

    Parameters:
        model_pred: Model predictions on test set
        ground_truth: ground truth
        norm_values: normalization values
        average_over_T: If True: Compute the average PSNR for different t values over T.
                        If False: Compute overall PSNR
        normalize: Normalize 3D Volumes by dividing by the largest absolute value of the ground truth
                    Apply same normalization to the model output - then compute PSNR.

    Returns:
        PSNR, or averaged PSNR over T.
    """
    x, y, z, t, T = ground_truth.shape
    
    if normalize:
        max_abs_value_ground_truth =  norm_values
        ground_truth = ground_truth / max_abs_value_ground_truth
        model_pred = model_pred / max_abs_value_ground_truth
    
    #### PSNR makes sense for absolute value images
    model_pred = np.abs(model_pred)
    ground_truth = np.abs(ground_truth)
    
    # Compute the Mean Squared Error (MSE)
    mse = np.mean((model_pred - ground_truth)**2, axis=(0, 1, 2))
    
    # Compute the maximum possible pixel intensity squared
    max_signal_per_spatial_volume = np.max(np.abs(ground_truth), axis=(0, 1, 2))**2
    
    # Compute PSNR
    psnr_per_spatial_volume = 10 * np.log10(max_signal_per_spatial_volume / mse)
    
    if average_over_T == False:
        final_psnr = np.mean(psnr_per_spatial_volume) # Overall PSNR
        return final_psnr
    
    else:
        psnr_vector = np.mean(psnr_per_spatial_volume, axis=1) # Averages over T
        return psnr_vector

def PSNR_spectral_domain(model_pred, ground_truth, norm_values, average_over_T = False, normalize = False):
    """
    Compute PSNR for model predictions in spectral domain (has to be in format x,y,z,t,T).
    The model_pred is automatically 0 filled along t to match ground_truth in case they have
    divergent shape.

    Parameters:
        model_pred: Model predictions on test set
        ground_truth: ground truth - dont use truncated t version for spectral domain
        norm_values: normalization values
        average_over_T: If True: Compute the average PSNR for different t values over T.
                        If False: Compute overall PSNR
        normalize: Normalize in the spectral domain (divide by maximum absolute value in spectral domain)

    Returns:
        PSNR in spectral domain, or averaged PSNR over T.
    """
    t_ground_truth = ground_truth.shape[3]
    t_model =  model_pred.shape[3]
    
    if t_model < t_ground_truth: # apply zero filling
        extended_output = np.zeros(ground_truth.shape, dtype=model_pred.dtype)
        extended_output[..., :t_model, :] =  model_pred
        model_pred = extended_output
        
    ground_truth_spectral = np.fft.fftshift(np.fft.fft(ground_truth, axis=-2), axes=-2)
    model_pred_spectral = np.fft.fftshift(np.fft.fft(model_pred, axis=-2), axes=-2)
    
    psnr = PSNR_time_domain(model_pred_spectral, ground_truth_spectral, norm_values, average_over_T = average_over_T, normalize = normalize)
    
    return psnr

def plot_general_statistics_PSNR(model_pred, model_input, full_ground_truth, truncate_t, norm_values, label = "Model Output", label2 = "IFFT"):
    """
    This function creates a single row of 2 plots with two series each:
        1. PSNR in time domain (non-normalized)
        2. PSNR in spectral domain (without normalization)

    For each plot:
       - One curve for 'model_pred' (network output)
       - One curve for 'model_input' (network input)

    Parameters:
        model_pred:    Model predictions on test set
        model_input:   Network input (undersampled or partial data)
        full_ground_truth: Ground truth - don't use truncated t version for spectral domain
        truncate_t:    How many time steps t have been used for the network
        norm_values:   Normalization values for time domain
        label:         Change label of the middle column if needed

    Returns:
        A single row of 2 plots comparing PSNR of 'model_pred' vs. 'model_input'
    """
    # Match shapes of ground truth and predictions for time domain
    ground_truth = full_ground_truth[:, :, :, :truncate_t, :]

    #### Compute PSNR for model_pred (Output) in time domain ####
    psnr_time_non_normalized_pred = PSNR_time_domain(
        model_pred, ground_truth, norm_values, average_over_T=True, normalize=False
    )
    psnr_time_non_normalized_total_pred = PSNR_time_domain(
        model_pred, ground_truth, norm_values, average_over_T=False, normalize=False
    )

    #### Compute PSNR for model_pred (Output) in spectral domain ####
    psnr_spectral_pred = PSNR_spectral_domain(
        model_pred, full_ground_truth, norm_values, average_over_T=True, normalize=False
    )
    psnr_spectral_total_pred = PSNR_spectral_domain(
        model_pred, full_ground_truth, norm_values, average_over_T=False, normalize=False
    )

    #### Compute PSNR for model_input (Input) in time domain ####
    psnr_time_non_normalized_inp = PSNR_time_domain(
        model_input, ground_truth, norm_values, average_over_T=True, normalize=False
    )
    psnr_time_non_normalized_total_inp = PSNR_time_domain(
        model_input, ground_truth, norm_values, average_over_T=False, normalize=False
    )

    #### Compute PSNR for model_input (Input) in spectral domain ####
    psnr_spectral_inp = PSNR_spectral_domain(
        model_input, full_ground_truth, norm_values, average_over_T=True, normalize=False
    )
    psnr_spectral_total_inp = PSNR_spectral_domain(
        model_input, full_ground_truth, norm_values, average_over_T=False, normalize=False
    )

    # Prepare the figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("PSNR in Time and Spectral Domains", fontsize=16)

    # Plot 1: PSNR in time domain (non-normalized)
    axes[0].plot(psnr_time_non_normalized_pred, marker='o', linestyle='-', label=label)
    axes[0].plot(psnr_time_non_normalized_inp,  marker='x', linestyle='--', label=label2)
    axes[0].set_title("Time Domain (Non-Normalized)")
    axes[0].set_xlabel("t Index")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].grid(True)
    axes[0].legend()

    # Plot 2: PSNR in spectral domain (no normalization)
    axes[1].plot(psnr_spectral_pred, marker='o', linestyle='-', label=label)
    axes[1].plot(psnr_spectral_inp,  marker='x', linestyle='--', label=label2)
    axes[1].set_title("Spectral Domain")
    axes[1].set_xlabel("Spectral Index")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].grid(True)
    axes[1].legend()

    # Adjust layout and show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # Print total PSNR values for both model_pred and model_input
    print("==== Model Output vs. Ground Truth ====")
    print(f"Average PSNR in image domain:    {psnr_time_non_normalized_total_pred}")
    print(f"Average PSNR in frequency domain: {psnr_spectral_total_pred}\n")

    print("==== Model Input vs. Ground Truth ====")
    print(f"Average PSNR in image domain:    {psnr_time_non_normalized_total_inp}")
    print(f"Average PSNR in frequency domain: {psnr_spectral_total_inp}")
    
def SSIM_time_domain(model_pred, ground_truth, norm_values, average_over_T=False, normalize=False):
    """
    Compute SSIM for model predictions (has to be in format x,y,z,t,T).

    Parameters:
        model_pred: Model predictions on test set
        ground_truth: Ground truth
        norm_values: Normalization values (peak values for each spatial volume)
        average_over_T: If True: Compute the average SSIM for different t values over T.
                        If False: Compute overall SSIM
        normalize: Normalize 3D Volumes by dividing by the largest absolute value of the ground truth
                   Apply the same normalization to the model output - then compute SSIM.

    Returns:
        SSIM, or averaged SSIM over T.
    """
    x, y, z, t, T = ground_truth.shape

    if normalize:
        max_abs_value_ground_truth = norm_values
        ground_truth = ground_truth / max_abs_value_ground_truth
        model_pred = model_pred / max_abs_value_ground_truth

    #### SSIM makes sense for absolute value images
    model_pred = np.abs(model_pred)
    ground_truth = np.abs(ground_truth)

    ssim_per_spatial_volume = []

    # Compute SSIM slice-by-slice for 3D volumes
    for t_idx in range(t):
        for T_idx in range(T):
            ssim_value = ssim(
                model_pred[:, :, :, t_idx, T_idx],
                ground_truth[:, :, :, t_idx, T_idx],
                data_range=ground_truth[:, :, :, t_idx, T_idx].max() - ground_truth[:, :, :, t_idx, T_idx].min()
            )
            ssim_per_spatial_volume.append(ssim_value)

    ssim_per_spatial_volume = np.array(ssim_per_spatial_volume).reshape(t, T)

    if average_over_T == False:
        # Compute the overall SSIM across all dimensions
        final_ssim = np.mean(ssim_per_spatial_volume)  # Overall SSIM
        return final_ssim
    else:
        # Compute SSIM averaged over T
        ssim_vector = np.mean(ssim_per_spatial_volume, axis=1)  # Averages over T
        return ssim_vector


def SSIM_spectral_domain(model_pred, ground_truth, norm_values, average_over_T=False, normalize=False):
    """
    Compute SSIM for model predictions in spectral domain (has to be in format x,y,z,t,T).
    The model_pred is automatically 0 filled along t to match ground_truth in case they have
    divergent shape.

    Parameters:
        model_pred: Model predictions on test set
        ground_truth: Ground truth - don't use truncated t version for spectral domain
        norm_values: Normalization values
        average_over_T: If True: Compute the average SSIM for different t values over T.
                        If False: Compute overall SSIM
        normalize: Normalize in the spectral domain (divide by maximum absolute value in spectral domain)

    Returns:
        SSIM in spectral domain, or averaged SSIM over T.
    """
    t_ground_truth = ground_truth.shape[3]
    t_model = model_pred.shape[3]

    if t_model < t_ground_truth:  # Apply zero filling
        extended_output = np.zeros(ground_truth.shape, dtype=model_pred.dtype)
        extended_output[..., :t_model, :] = model_pred
        model_pred = extended_output

    ground_truth_spectral = np.fft.fftshift(np.fft.fft(ground_truth, axis=-2), axes=-2)
    model_pred_spectral = np.fft.fftshift(np.fft.fft(model_pred, axis=-2), axes=-2)

    ssim_value = SSIM_time_domain(model_pred_spectral, ground_truth_spectral, norm_values,
                                  average_over_T=average_over_T, normalize=normalize)
    return ssim_value


def plot_general_statistics_SSIM(model_pred, model_input, full_ground_truth, truncate_t, norm_values, label = "Model Output", label2 = "IFFT"):
    """
    This function creates a single row of 2 plots with two series each:
        1. SSIM in time domain
        2. SSIM in spectral domain

    For each plot:
       - One curve for 'model_pred' (network output)
       - One curve for 'model_input' (network input)

    Parameters:
        model_pred:    Model predictions on test set
        model_input:   Network input (undersampled or partial data)
        full_ground_truth: Ground truth - don't use truncated t version for spectral domain
        truncate_t:    How many time steps t have been used for the network
        norm_values:   Normalization values for time domain
        label:         Change label of the middle column if needed

    Returns:
        A single row of 2 plots comparing SSIM of 'model_pred' vs. 'model_input'
    """
    # Match shapes of ground truth and predictions for time domain
    ground_truth = full_ground_truth[:, :, :, :truncate_t, :]

    #### Compute SSIM for model_pred (Output) in time domain ####
    ssim_time_pred = SSIM_time_domain(
        model_pred, ground_truth, norm_values, average_over_T=True, normalize=False
    )
    ssim_time_total_pred = SSIM_time_domain(
        model_pred, ground_truth, norm_values, average_over_T=False, normalize=False
    )

    #### Compute SSIM for model_pred (Output) in spectral domain ####
    ssim_spectral_pred = SSIM_spectral_domain(
        model_pred, full_ground_truth, norm_values, average_over_T=True, normalize=False
    )
    ssim_spectral_total_pred = SSIM_spectral_domain(
        model_pred, full_ground_truth, norm_values, average_over_T=False, normalize=False
    )

    #### Compute SSIM for model_input (Input) in time domain ####
    ssim_time_inp = SSIM_time_domain(
        model_input, ground_truth, norm_values, average_over_T=True, normalize=False
    )
    ssim_time_total_inp = SSIM_time_domain(
        model_input, ground_truth, norm_values, average_over_T=False, normalize=False
    )

    #### Compute SSIM for model_input (Input) in spectral domain ####
    ssim_spectral_inp = SSIM_spectral_domain(
        model_input, full_ground_truth, norm_values, average_over_T=True, normalize=False
    )
    ssim_spectral_total_inp = SSIM_spectral_domain(
        model_input, full_ground_truth, norm_values, average_over_T=False, normalize=False
    )

    # Prepare the figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("SSIM in Time and Spectral Domains", fontsize=16)

    # Plot 1: SSIM in time domain
    axes[0].plot(ssim_time_pred, marker='o', linestyle='-', label=label)
    axes[0].plot(ssim_time_inp,  marker='x', linestyle='--', label=label2)
    axes[0].set_title("Time Domain")
    axes[0].set_xlabel("t Index")
    axes[0].set_ylabel("SSIM")
    axes[0].grid(True)
    axes[0].legend()

    # Plot 2: SSIM in spectral domain
    axes[1].plot(ssim_spectral_pred, marker='o', linestyle='-', label=label)
    axes[1].plot(ssim_spectral_inp,  marker='x', linestyle='--', label=label2)
    axes[1].set_title("Spectral Domain")
    axes[1].set_xlabel("Spectral Index")
    axes[1].set_ylabel("SSIM")
    axes[1].grid(True)
    axes[1].legend()

    # Adjust layout and show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # Print total SSIM values for both model_pred and model_input
    print("==== Model Output vs. Ground Truth ====")
    print(f"Average SSIM in image domain:    {ssim_time_total_pred}")
    print(f"Average SSIM in frequency domain: {ssim_spectral_total_pred}\n")

    print("==== Model Input vs. Ground Truth ====")
    print(f"Average SSIM in image domain:    {ssim_time_total_inp}")
    print(f"Average SSIM in frequency domain: {ssim_spectral_total_inp}")
    
def generate_noise_volume(shape, variance):
    """
    Generate a 3D volume of complex Gaussian noise.
    
    Parameters:
    - shape (tuple): The shape of the 3D volume (Nx, Ny, Nz).
    - variance (float): The variance of the Gaussian noise.
    
    Returns:
    - noise_volume (numpy.ndarray): A complex-valued 3D array with simulated noise.
    """
    # Generate real and imaginary components of Gaussian noise
    real_noise = np.random.normal(loc=0.0, scale=np.sqrt(variance), size=shape)
    imag_noise = np.random.normal(loc=0.0, scale=np.sqrt(variance), size=shape)
    
    # Create complex noise
    noise_volume = real_noise + 1j * imag_noise
    
    return noise_volume

def estimate_noise_variance(noise_data):
    """
    Estimate the noise variance from a dataset that contains only noise.

    Parameters:
    - noise_data (numpy.ndarray): A complex-valued array of noise-only data.

    Returns:
    - variance_real (float): Estimated variance of the real part.
    - variance_imag (float): Estimated variance of the imaginary part.
    - total_variance (float): Average variance across real and imaginary components.
    """
    real_part = noise_data.real
    imag_part = noise_data.imag

    variance_real = np.var(real_part, ddof=1)  # Use ddof=1 for unbiased estimate
    variance_imag = np.var(imag_part, ddof=1)

    total_variance = (variance_real + variance_imag) / 2  # Mean variance

    return variance_real, variance_imag, total_variance

import torch
import numpy as np

def evaluate_model_for_predictions(model, test_loader, device, original_shape, inverse_reshape):
    """
    Runs inference on the given model and test_loader,
    and returns only the final 'predictions' array.
    """
    model.eval()

    # List to accumulate outputs
    outputs_list = []

    # Disable gradient computation for efficiency
    with torch.no_grad():
        for data, _ in test_loader:  # FIX: 'data_loader' to 'test_loader'
            # Move input to device
            inputs_img = data.to(device)

            # Forward pass (as a tuple if your model expects a tuple)
            outputs = model(inputs_img)  # FIX: Removed unnecessary extra parentheses

            # If outputs is a tuple, extract the first element
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # Store output in CPU numpy array
            outputs_list.append(outputs.cpu().numpy())

    # Concatenate all outputs
    outputs_array = np.concatenate(outputs_list, axis=0)
    outputs_array_complex = outputs_array[:, 0, :, :] + 1J * outputs_array[:, 1, :, :]

    # Reorder dimensions
    outputs_array_complex_transpose = outputs_array_complex.transpose(2, 1, 0)

    # Reshape and transpose using inverse_reshape
    outputs_array_reshaped = outputs_array_complex_transpose.reshape(original_shape)
    Model_Output = outputs_array_reshaped.transpose(inverse_reshape)

    return Model_Output


def evaluate_model_for_predictions_TEST(model, test_loader, device):
    """
    Runs inference on the given model and test_loader,
    and returns only the final 'predictions' array.
    """
    model.eval()

    # List to accumulate outputs
    outputs_list = []

    # Disable gradient computation for efficiency
    with torch.no_grad():
        for data, _ in test_loader:
            # Move input to device
            inputs_img = data.to(device)
            
            # Forward pass (as a tuple if your model expects a tuple)
            outputs = model((inputs_img))

            # If outputs is a tuple, extract the first element
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # Store output in CPU numpy array
            outputs_list.append(outputs.cpu().numpy())

    # Concatenate all outputs
    outputs_array = np.concatenate(outputs_list, axis=0)

    # Extract real and imaginary parts
    outputs_array_real = outputs_array[:, 0, :, :]
    outputs_array_imag = outputs_array[:, 1, :, :]
    outputs_array_complex = outputs_array_real + 1j * outputs_array_imag

    # Reorder dimensions
    outputs_array_complex_transpose = outputs_array_complex.transpose(1, 2, 0)

    # Reshape and transpose to desired shape
    predictions_wrong_shape = outputs_array_complex_transpose.reshape(22, 22, 21, truncate_t, 8)
    predictions = predictions_wrong_shape.transpose(1, 0, 2, 3, 4)

    return predictions

def evaluate_Unet(model, data_loader, device, inverse_reshape_for_pytorch, norm_values, original_shape, inverse_reshape, grouped_time_steps):
    """
    Evaluates the model on a given data loader and processes the output.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader for evaluation data.
        device (torch.device): Device to run evaluation on (CPU or GPU).
        inverse_reshape_for_pytorch (function): Function to reshape output back.
        norm_values (np.ndarray): Normalization values for denormalization.
        original_shape (tuple): The original shape of the data before processing.
        inverse_reshape (tuple): Tuple describing the inverse reshaping order.
        grouped_time_steps (int): Time steps grouped in the input.

    Returns:
        np.ndarray: Model output in its final processed form.
    """
    model.eval()  # Set the model to evaluation mode

    outputs_list = []

    # Disable gradient computation for efficiency
    with torch.no_grad():
        for data, _, _ in data_loader:
            inputs_img = data.to(device)  # Move input tensor to the appropriate device
            outputs = model((inputs_img))  # Pass the inputs as a tuple to the model

            # If model returns a tuple, extract the first element
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            outputs_list.append(outputs.cpu().numpy())  # Convert to numpy and move to CPU

    # Concatenate all outputs
    outputs_array = np.concatenate(outputs_list, axis=0)

    # Reshape for PyTorch processing
    outputs_array = inverse_reshape_for_pytorch(outputs_array, grouped_time_steps)

    # Denormalize the data
    denormalized_output = denormalize_data_per_image(outputs_array, norm_values.reshape(-1))

    # Restore original shape
    denormalized_output = denormalized_output.reshape(original_shape)

    # Apply final transformation
    Model_Output = denormalized_output.transpose(inverse_reshape)

    # Crop to match expected test set shape
    Model_Outputs_Test_Set = Model_Output[1:-1, 1:-1, 1:-2, ...]

    return Model_Outputs_Test_Set
    
def evaluate_WNet(
    model,
    data_loader,
    device,
    inverse_reshape_for_pytorch,
    norm_values,
    original_shape,
    inverse_reshape,
    grouped_time_steps
):
    """
    Evaluates the WNet on a given data loader and processes the output.

    Args:
        model (torch.nn.Module): The trained WNet model to evaluate (with 3 encoders).
        data_loader (torch.utils.data.DataLoader): DataLoader for evaluation data.
        device (torch.device): Device to run evaluation on (CPU or GPU).
        inverse_reshape_for_pytorch (function): Function to reshape output back
            to the original pre-training format (inverse of 'reshape_for_pytorch').
        norm_values (np.ndarray): Normalization values for denormalizing.
        original_shape (tuple): The original shape of the data before processing.
        inverse_reshape (tuple): Tuple describing the inverse transposing order.
        grouped_time_steps (int): The number of time steps grouped in the input.

    Returns:
        np.ndarray: Model output in its final processed form (e.g. cropped test set).
    """
    model.eval()  # Set the model to evaluation mode

    outputs_list = []

    with torch.no_grad():
        # Iterate over the evaluation DataLoader
        for data, _, _ in data_loader:
            # 'data' shape is typically (batch_size, 2, z, f, T)
            data = data.to(device)

            # 1) Split into three spectral windows for the WNet branches
            #    Adjust idxA, idxB, idxC as needed for your spectral ranges
            xA, xB, xC = split_spectrum_into_3(data,
                                               idxA=(0,55),
                                               idxB=(56,71),
                                               idxC=(72,None))
            xA, xB, xC = xA.to(device), xB.to(device), xC.to(device)

            # 2) Forward pass through the three-branch model
            outputs = model(xA, xB, xC)  # shape: (batch_size, 2, z, f, T) or similar

            # If your model returns a tuple, extract the first element
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # 3) Move outputs to CPU and store as numpy
            outputs_list.append(outputs.cpu().numpy())

    # 4) Concatenate all batch outputs along the batch dimension
    outputs_array = np.concatenate(outputs_list, axis=0)
    # Now shape is (N, 2, z, f, T) or similar, for N = total # of samples

    # 5) "Inverse reshape" to undo the shape transformations
    outputs_array = inverse_reshape_for_pytorch(outputs_array, grouped_time_steps)

    # 6) Denormalize if you used per-image normalization
    denormalized_output = denormalize_data_per_image(outputs_array, norm_values.reshape(-1))

    # 7) Restore the original shape (before data prep)
    denormalized_output = denormalized_output.reshape(original_shape)

    # 8) Transpose back to your desired dimension order
    Model_Output = denormalized_output.transpose(inverse_reshape)

    # 9) Crop any padded edges if needed (example slice)
    #    Adjust as needed for your actual padding
    Model_Outputs_Test_Set = Model_Output[1:-1, 1:-1, 1:-2, ...]

    return Model_Outputs_Test_Set

def evaluate_Ynet(model, data_loader, device, inverse_reshape_for_pytorch, 
                  norm_values, original_shape, inverse_reshape, grouped_time_steps):
    """
    Evaluates the two-branch YNet on a given data loader and processes the output.

    Args:
        model (torch.nn.Module): The trained YNet model.
        data_loader (torch.utils.data.DataLoader): DataLoader yielding batches of 
            (data, targets, mask, gm_mask), where:
            - data: shape (N, 2, z, f, T)
            - targets: same shape as data
            - mask: overall mask of shape (N, 2, z, f, T)
            - gm_mask: gray matter mask of shape (N, 1, z, f, T)
        device (torch.device): Device to run evaluation on.
        inverse_reshape_for_pytorch (function): Function to reshape output back.
        norm_values (np.ndarray): Normalization values for denormalization.
        original_shape (tuple): The original shape of the data before processing.
        inverse_reshape (tuple): Tuple describing the inverse transposing order.
        grouped_time_steps (int): Number of time steps grouped in the input.

    Returns:
        np.ndarray: Final processed model output (e.g., cropped for test set).
    """
    model.eval()  # Set model to evaluation mode
    outputs_list = []

    with torch.no_grad():
        for data, targets, mask, gm_mask in data_loader:
            # Move data and masks to device
            data    = data.to(device)           # (N, 2, z, f, T)
            mask    = mask.to(device)           # (N, 2, z, f, T)
            gm_mask = gm_mask.to(device)         # (N, 1, z, f, T)
            
            # Create two branch inputs:
            # Gray matter branch: only GM regions remain.
            xGM = data * gm_mask
            # White matter branch: use the complementary mask (assuming gm_mask + wm_mask = 1).
            xWM = data * (1.0 - gm_mask)
            
            # Forward pass through YNet
            outputs = model(xGM, xWM)  # Expected shape: (N, 2, z, f, T)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            outputs_list.append(outputs.cpu().numpy())

    # Concatenate outputs from all batches along the first axis
    outputs_array = np.concatenate(outputs_list, axis=0)

    # Inverse reshape to undo previous pytorch reshaping
    outputs_array = inverse_reshape_for_pytorch(outputs_array, grouped_time_steps)

    # Denormalize the data using provided norm_values
    denormalized_output = denormalize_data_per_image(outputs_array, norm_values.reshape(-1))

    # Restore the original shape of the data
    denormalized_output = denormalized_output.reshape(original_shape)

    # Apply final transpose (inverse_reshape) to get desired output order
    Model_Output = denormalized_output.transpose(inverse_reshape)

    # Crop the output if necessary to match the expected test-set shape
    Model_Outputs_Test_Set = Model_Output[1:-1, 1:-1, 1:-2, ...]

    return Model_Outputs_Test_Set

def evaluate_Unet_masked(model, data_loader, device, inverse_reshape_for_pytorch, 
                         norm_values, original_shape, inverse_reshape, grouped_time_steps):
    """
    Evaluates the updated UNet (with extra mask channels) on a given data loader 
    and processes the output.
    
    Args:
        model (torch.nn.Module): The trained model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader yielding tuples:
            (dat_in, dat_out, mask, gm_mask, wm_mask)
            - dat_in: spectral data, shape (N, 2*k, H, W, D)
            - dat_out: ground truth, same shape as dat_in.
            - mask: overall mask (N, 2, H, W, D)
            - gm_mask: gray matter mask (N, 1, H, W, D)
            - wm_mask: white matter mask (N, 1, H, W, D)
        device (torch.device): Device to run evaluation on (CPU or GPU).
        inverse_reshape_for_pytorch (function): Function to invert the reshaping done for PyTorch.
        norm_values (np.ndarray): Normalization values for denormalization.
        original_shape (tuple): The original shape of the data before processing.
        inverse_reshape (tuple): Tuple describing the inverse transposing order.
        grouped_time_steps (int): Number of time steps grouped in the input.
    
    Returns:
        np.ndarray: Final processed model output (e.g., cropped to match expected test set shape).
    """
    model.eval()  # Set model to evaluation mode

    outputs_list = []

    # Disable gradient computation for efficiency
    with torch.no_grad():
        for dat_in, dat_out, mask, gm_mask, wm_mask in data_loader:
            # Move inputs, outputs, and masks to the specified device
            dat_in  = dat_in.to(device)    # shape: (N, 2*k, H, W, D)
            mask    = mask.to(device)      # shape: (N, 2, H, W, D)
            gm_mask = gm_mask.to(device)    # shape: (N, 1, H, W, D)
            wm_mask = wm_mask.to(device)    # shape: (N, 1, H, W, D)
            
            # Concatenate spectral data with the extra masks to form input of shape (N, 2*k+2, H, W, D)
            inputs_img = torch.cat([dat_in, gm_mask, wm_mask], dim=1)
            
            # Forward pass
            outputs = model(inputs_img)  # expected output shape: (N, 2*k, H, W, D)
            
            # If model returns a tuple, extract the first element
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            outputs_list.append(outputs.cpu().numpy())

    # Concatenate all outputs along the batch dimension
    outputs_array = np.concatenate(outputs_list, axis=0)

    # Inverse reshape to undo PyTorch processing
    outputs_array = inverse_reshape_for_pytorch(outputs_array, grouped_time_steps)

    # Denormalize the output using the provided normalization values
    denormalized_output = denormalize_data_per_image(outputs_array, norm_values.reshape(-1))

    # Restore the original shape
    denormalized_output = denormalized_output.reshape(original_shape)

    # Apply the final transpose to restore the original ordering
    Model_Output = denormalized_output.transpose(inverse_reshape)

    # Crop to match expected test set shape (adjust the slicing as needed)
    Model_Outputs_Test_Set = Model_Output[1:-1, 1:-1, 1:-2, ...]

    return Model_Outputs_Test_Set

    