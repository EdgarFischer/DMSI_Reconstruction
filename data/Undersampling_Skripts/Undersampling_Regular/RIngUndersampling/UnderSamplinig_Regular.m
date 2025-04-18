%%   
AF = 4; % define acceleration factor

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

    % note that technically P03 had a 44th ring, but this makes ~0.6 percent of
    % extra kspace data which is why I just keep it, it makes a negligible
    % difference
    undersamplingPattern = load(sprintf('csiUndersampled_AF%d.mat', AF));

    USPattern = undersamplingPattern.csiUS;
    newMask = USPattern.Data;

    %%
    % Undersampling
    % csi = csi()...
    % Assume USPattern and csikspace are already loaded.

    % Loop over each ring
    % Assume:
    % csikspace.Data is a cell array where each cell contains the k-space data for a ring.
    % Each k-space data array has dimensions:
    % [526, 1, numKz, 1, 96, 1, numTime]
    %
    % newMask is a [numRings x numTime] matrix with binary values.

    numRings = numel(csikspace.Data);
    for i = 1:numRings
        % Get the k-space data for the current ring.
        kspaceData = csikspace.Data{i};

        % Determine the number of time frames from the 7th dimension.
        numTime = size(kspaceData, 7);

        % Loop over each time frame.
        for j = 1:numTime
            % Check if the mask for ring i at time j is 0.
            if newMask(i, j) == 0
                % If it is, set all k-space data for that time frame to 0.
                kspaceData(:, 1, :, 1, :, 1, j) = 0;
            end
        end

        % Update the cell with the modified k-space data.
        csikspace.Data{i} = kspaceData;
    end






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

