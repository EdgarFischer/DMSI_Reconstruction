%%   
AF = 5; % define acceleration factor
Patient = 3;

    
%Patient = 3; % allowed 3-8

% the rest is automatic only the Patient variable needs to be adjusted

folderPath_P03 = '/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/lab/Measurement_Data/7T_DMI/fn_vb_DMI_CRT_P03/meas_MID00036_FID137531_fn_3D_DW_DMI_9_1mm_54min.dat';
folderPath_P04 = '/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/lab/Measurement_Data/7T_DMI/fn_vb_DMI_CRT_P04/meas_MID00032_FID138303_fn_3D_DW_DMI_9_1_290TR_56min.dat';
folderPath_P05 = '/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/lab/Measurement_Data/7T_DMI/fn_vb_DMI_CRT_P05/meas_MID00033_FID138510_fn_3D_DW_DMI_9_1_290TR_56min.dat';
folderPath_P06 = '/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/lab/Measurement_Data/7T_DMI/fn_vb_DMI_CRT_P06/meas_MID00032_FID140145_fn_3D_DW_DMI_9_1_290TR_56min.dat';
folderPath_P07 = '/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/lab/Measurement_Data/7T_DMI/fn_vb_DMI_CRT_P07/meas_MID00034_FID140197_fn_3D_DW_DMI_9_1_290TR_56min.dat';
folderPath_P08 = '/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/lab/Measurement_Data/7T_DMI/fn_vb_DMI_CRT_P08/meas_MID00033_FID142160_fn_3D_DW_DMI_9_1_290TR_56min.dat';

% Map patient number to folder path
folderPaths = containers.Map({3,4,5,6,7,8}, ...
                             {folderPath_P03, folderPath_P04, folderPath_P05, ...
                              folderPath_P06, folderPath_P07, folderPath_P08});

% Select the correct folder based on the Patient variable
if isKey(folderPaths, Patient)
    selectedFolder = folderPaths(Patient);
else
    error('Invalid Patient number. Choose a number between 3 and 8.');
end

which op_AverageMRData
% rmpath(genpath())
addpath(genpath('/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/hfish/Part1_Reco_LCModel_v2.0.0_DW_CRT_multiple_reps/MatlabFunctions'))
Par.Paths.out_path = '';

%%
% Path P03: 
%testa

[csikspace, image, NoiseData] = io_ReadAndReshapeSiemensData(folderPaths(Patient));


%%
% if(isfield(NoiseData,'Data') && numel(NoiseData.Data) > 1 && Par.Flags.noisedecorrelation_flag)
%     [NoiseCorrMatStruct,Dummy] = op_CalcNoiseCorrMat(NoiseData);
%     NoiseData = Dummy.NoiseData; clear Dummy;
% end



%%
csikspace = op_AverageMRData(csikspace);

Data = csikspace.Data;

% Maximum number of k_z indices (always 43)
maxKz = 43;
% Number of rings (cells in Data)
numRings = numel(Data);
% Preallocate the output (rows = rings, columns = k_z indices)
result = zeros(numRings, maxKz);

% Define the central column (this will always correspond to k_z = 0)
centerColumn = ceil(maxKz/2);  % For 43, centerColumn = 22

% Loop over each ring
for i = 1:numRings
    % Get current k_z count for ring i (3rd dimension of Data{i})
    currentKz = size(Data{i}, 3);
    % Find the middle index for this ring's k_z values (which is k_z = 0)
    midIndex = ceil(currentKz/2);
    
    % Loop over available k_z indices for this ring
    for k = 1:currentKz
        % Map the local index k to a global column:
        % shift so that when k == midIndex, column == centerColumn
        col = centerColumn + (k - midIndex);
        
        % Only assign if the computed column is within the final matrix bounds
        if col >= 1 && col <= maxKz
            % Extract the value from Data{i} with the other dimensions fixed at index 1.
            result(i, col) = Data{i}(1, 1, k, 1, 2, 1, 8);
        end
    end
end

% Compute the absolute value of result (adding eps if needed to avoid log(0))
log_result = abs(result);

% Create a new figure for the plot
figure;
imagesc(log_result');  % Transpose to swap axes
%caxis([-5 -4]);  % Optionally fix the color scale from -6 to -3
colorbar;
xlabel('Ring index');   % Now x-axis corresponds to the ring index
ylabel('k_z index');    % Now y-axis corresponds to the k_z index
title('2D Plot of |signal|');




