%%   
AF = 5; % define acceleration factor


    
Patient = 3; % allowed 3-8

% the rest is automatic only the Patient variable needs to be adjusted

folderPath_P03 = '/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/lab/Measurement_Data/7T_DMI/fn_vb_DMI_CRT_P03/meas_MID00036_FID137531_fn_3D_DW_DMI_9_1mm_54min.dat';
folderPath_P04 = '/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/lab/Measurement_Data/7T_DMI/fn_vb_DMI_CRT_P04/meas_MID00032_FID138303_fn_3D_DW_DMI_9_1_290TR_56min.dat';
folderPath_P05 = '/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/lab/Measurement_Data/7T_DMI/fn_vb_DMI_CRT_P05/meas_MID00033_FID138510_fn_3D_DW_DMI_9_1_290TR_56min.dat';
folderPath_P06 = '/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/lab/Measurement_Data/7T_DMI/fn_vb_DMI_CRT_P06/meas_MID00032_FID140145_fn_3D_DW_DMI_9_1_290TR_56min.dat';
folderPath_P07 = '/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/lab/Measurement_Data/7T_DMI/fn_vb_DMI_CRT_P07/meas_MID00034_FID140197_fn_3D_DW_DMI_9_1_290TR_56min.dat';
folderPath_P08 = '/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/lab/Measurement_Data/7T_DMI/fn_vb_DMI_CRT_P08/meas_MID00033_FID142160_fn_3D_DW_DMI_9_1_290TR_56min.dat';

Multi_Channel = '/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/lab/Measurement_Data/7T_DMI/Torso_2H/fn_WB_bow_b0_shim/part2/meas_MID00161_FID32932_fn_bs_dw_crt_240627_tuneup_32x32_1200V_3.dat';

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

[csikspace, image, NoiseData] = io_ReadAndReshapeSiemensData(Multi_Channel);


%%
% if(isfield(NoiseData,'Data') && numel(NoiseData.Data) > 1 && Par.Flags.noisedecorrelation_flag)
%     [NoiseCorrMatStruct,Dummy] = op_CalcNoiseCorrMat(NoiseData);
%     NoiseData = Dummy.NoiseData; clear Dummy;
% end

csikspace = op_AverageMRData(csikspace);

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
    filename2 = 'Multi_Channel.mat';
    save(filename2, 'csi')







