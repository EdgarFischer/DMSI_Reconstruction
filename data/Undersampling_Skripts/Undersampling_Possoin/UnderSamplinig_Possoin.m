%%   
%%   
AF = 8; % define acceleration factor

for Patient = 3:8
    
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

    if Patient == 3 %had more rings, therefore different undersampling
        undersamplingPattern = load(sprintf('csiUndersampled_AF%d_P03.mat', AF));
    else
        undersamplingPattern = load(sprintf('csiUndersampled_AF%d.mat', AF));
    end


    USPattern = undersamplingPattern.csiUS;

    %%
    % Undersampling
    % csi = csi()...
    % Assume USPattern and csikspace are already loaded.

    % Loop over each ring
    for i = 1:numel(csikspace.Data)
        % Get the complete k-space data for ring i.
        % This array has size: [526, 1, numKz, 1, 96, 1, numTime]
        kspaceData = csikspace.Data{i};  
        % Determine the number of kz indices and time frames
        numKz = size(kspaceData, 3);
        numTime = size(kspaceData, 7);

        % Loop over each time point
        for j = 1:numTime
            % Get the list of kz indices that should be sampled for this ring and time
            validKz = USPattern.Data{i}{j};  % This should be a vector of valid indices

            % Loop over all kz indices
            for kz = 1:numKz
                % Check if the current kz index is in the list of valid indices.
                if ~ismember(kz, validKz)
                    % If not, set the data for this kz index at time j to zero.
                    % We assume that the dimensions are:
                    % 1: 526 (all samples along the readout)
                    % 2: fixed (1)
                    % 3: kz index (which we are iterating over)
                    % 4: fixed (1)
                    % 5: another dimension (here all 96 elements)
                    % 6: fixed (1)
                    % 7: temporal index (j)
                    kspaceData(:, 1, kz, 1, :, 1, j) = 0;
                end
            end
        end

        % Store the modified data back into the structure.
        csikspace.Data{i} = kspaceData;
    end
    
    % Calculate the percentage of zeros in csikspace.Data
    totalZeros = 0;
    totalElements = 0;

    for i = 1:numel(csikspace.Data)
        currentData = csikspace.Data{i};
        totalZeros = totalZeros + sum(currentData(:) == 0);
        totalElements = totalElements + numel(currentData);
    end

    percentageZeros = (totalZeros / totalElements) * 100;
    fprintf('Percentage of zeros: %.2f%%\n', percentageZeros);




    % Reco of undersampling



    %%
    % if(exist('NoiseCorrMatStruct','var'))
    %     csi = op_PerformNoiseDecorrelation(csi,NoiseCorrMatStruct);
    %     if(exist('image','var'))
    %         image = op_PerformNoiseDecorrelation(image,NoiseCorrMatStruct);
    %     end
    % end


    if(isfield(csikspace.Par,'Hamming_flag') && csikspace.Par.Hamming_flag)
        Settings.NonCartReco.DensComp_flag = false;
    end
    Settings.NonCartReco.DensComp.Method = 'ConcentricRingTrajectory_Theoretical';    
    Settings.NonCartReco.ConjInkSpace_flag = true; 
    Settings.NonCartReco.FlipDim_flag = true;
    Settings.NonCartReco.Phaseroll_flag = true;
    Settings.NonCartReco.ConjIniSpace_flag = false; 

    % Reco MRSI Data
    %
    csi = op_ReconstructMRData(csikspace,struct(),Settings);

    csi.Data = csi.Data * 10^5;

    %%
    % Save Data
    filename2 = sprintf('Undersampled_P%02d_AF%d.mat', Patient, AF);
    save(filename2, 'csi')
end


