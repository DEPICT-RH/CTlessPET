import argparse
from typing import Union
from pathlib import Path, PosixPath
import warnings
import tempfile
import time
import subprocess as subp
from multiprocessing import Process
import onnxruntime
import torchio as tio
import numpy as np
from torch.utils.data import DataLoader
import torch
from pydicom import dcmread
from CTlessPET.utils import (
    maybe_download_weights,
    get_models,
    get_model_path,
    get_bed_path,
    percentile_norm
)
from CTlessPET.dicom_io import (
    get_sort_files_dict,
    to_dcm
)
import shutil
import dicom2nifti
from tqdm import tqdm
import nibabel as nib

#suppress warnings
warnings.filterwarnings('ignore')


class CTlessPET():
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.dose = None
        self.weight = None
        self.tracer = None
        
        self.NACCT_version = 'V0.1'
        
        
    def convert_dicom_data(self, input, CT):        
        self.tmp_dir_object = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp_dir_object.name)
        if self.verbose:
            print("\tCreated tmp folder", self.tmp_dir)
            
        if CT is None:
            # All dicom data is expected to be in "input" folder. We need to sort it.
            if self.verbose:
                print("\tFound both CT and NAC files in input folder. Sorting now")
                start_time = time.time()
                
            self.sorted_dicts = get_sort_files_dict(input)
            
            for modality, sorted_files in self.sorted_dicts.items():
                
                to_dir = Path(f'{self.tmp_dir}/{modality}')
                to_dir.mkdir(exist_ok=True, parents=True)
            
                for ind, p in enumerate(sorted_files.values()): 
                    shutil.copy(p, to_dir / f'{ind:04d}.dcm')
            
            if self.verbose:
                print(f'\tSorting files done in {time.time()-start_time:.01f} seconds')
            
            self.path_CT = self.tmp_dir / 'CT'
            self.path_NAC = self.tmp_dir / 'PT'
            
        else:
            # We expect input and CT to be two folders with dicom data in each. No need to sort, just convert.
            self.path_CT = CT
            self.path_NAC = input
            
            sorted_dict_CT = get_sort_files_dict(CT)
            sorted_dict_NAC = get_sort_files_dict(input)
            self.sorted_dicts = {'CT': sorted_dict_CT['CT'], 'PT': sorted_dict_NAC['PT']}
    
        # Read first NAC file and extract info
        d = dcmread(next(iter(self.sorted_dicts['PT'].values())))        
        weight = d.PatientWeight
        dose = d['RadiopharmaceuticalInformationSequence'][0]['RadionuclideTotalDose'].value / 1000000
        tracer = d['RadiopharmaceuticalInformationSequence'][0]['Radiopharmaceutical'].value
        self.set_dose_and_weight_and_tracer(dose=dose, weight=weight, tracer=tracer)
            
        if self.verbose:
            print("\tConverting files to nifti")
            start_time = time.time()
        path_CT_nii = self.tmp_dir / 'CT.nii.gz'
        path_NAC_nii = self.tmp_dir / 'NAC.nii.gz'
        dicom2nifti.dicom_series_to_nifti(self.path_CT, path_CT_nii, reorient_nifti=True)
        dicom2nifti.dicom_series_to_nifti(self.path_NAC, path_NAC_nii, reorient_nifti=True)
        if self.verbose:
            print(f'\tConverting files to nifti done in {time.time()-start_time:.01f} seconds')
            
        return path_NAC_nii, path_CT_nii
    

    def clean(self):
        self.tmp_dir_object.cleanup()
        

    # Sets up variables
    def setup(self, model): # Model = FDG, FDG_Pediatric, H2O
        
        # Get the type of model from the DICOM data if not already set
        if model is None:
            if self.tracer is None:
                raise ValueError('The choice for a model could not be inferred from the input data. Please specify.')
            elif self.tracer not in get_models()['Default']:
                raise ValueError(f'Not implemented for the tracer {self.tracer} yet')
            # TODO: Check for patient age to select children
            self.cohort = 'Default'
        elif model == 'FDG':
            self.cohort = 'Default'
            self.tracer = 'Fluorodeoxyglucose'
        elif model == 'FDG_Pediatric':
            self.cohort = 'Pediatric'
            self.tracer = 'Fluorodeoxyglucose'
        elif model == 'H2O':
            self.cohort = 'Default'
            self.tracer = 'Oxygen-water'    
        else:
            raise ValueError(f'Not implemented for the model {model} yet')

        # Get the model
        maybe_download_weights(self.cohort, self.tracer)
        if self.verbose:
            print("\tLoading model")
        self.model = onnxruntime.InferenceSession(
            get_model_path(self.cohort, self.tracer), 
            providers=['CUDAExecutionProvider'])
        if self.verbose:
            print(f"\tModel {self.cohort}_{self.tracer} loaded")

        # Patch details
        self.patch_size = [128,128,32]
        self.patch_overlap = (70,70,24)
        self.data_shape_in = [1] + self.patch_size
        
        self.input_name = self.model.get_inputs()[0].name
        
        
    # Overwrite dose and/or weight
    def set_dose_and_weight_and_tracer(self, dose=None, weight=None, tracer=None):
        if dose is not None:
            self.dose = dose
        if weight is not None:
            self.weight = weight
        if tracer is not None:
            self.tracer = tracer
            
            
    # Get mask of CT bed - requires CT is given
    def set_mask(self):
        
        pt = self.CT_subj
        
        # script_dir = os.path.dirname(os.path.abspath(__file__))
        mask = tio.ScalarImage(get_bed_path())

        # Computes difference in y
        y_diff = mask.affine[1,3]-pt.CT.affine[1,3]
        y_voxels = y_diff / pt.CT.spacing[1]
        y_diff,y_voxels

        # Computes difference in z
        z_diff = mask.affine[2,3]-pt.CT.affine[2,3]
        z_voxels = z_diff / pt.CT.spacing[2]
        z_diff,z_voxels

        # Cuts bed image in z
        z_start = np.abs(int(z_voxels))
        z_end = pt.CT.shape[-1]+z_start
        m = mask.data[:,:,:,z_start:z_end]

        # Shifts bed image in y
        m2 = np.zeros(m.shape)
        if (y_voxels > 0):  
            m2[:,:,int(y_voxels):,:] = m[:,:,:pt.shape[2]-int(y_voxels),:]
        else:
            m2[:,:,:pt.shape[2]-np.abs(int(y_voxels)),:] = m[:,:,np.abs(int(y_voxels)):,:]

        # Makes affine matrix for registered image
        aff = mask.affine.copy()
        aff[2,3] = pt.CT.affine[2,3].copy()
        aff[1,3] = pt.CT.affine[1,3].copy()

        self.bed_mask = tio.ScalarImage(tensor=m2, affine=aff)   
        
            
    # Preprocessing
    def preprocess(self, NAC, CT):
        
        self.NAC_path = NAC
        self.CT_path = CT
        
        # Load CT
        self.CT_subj = tio.Subject(CT=tio.ScalarImage(self.CT_path))
        
        # Compute BED mask from CT if set
        self.set_mask()

        # Resample CT to CT conform
        rsl_conform = tio.transforms.Resample((0.976562,0.976562,2))
        subj_conform = rsl_conform(self.CT_subj)

        # Cropping image to 512x512
        xy_original_ct = 512
        if xy_original_ct < subj_conform.CT.shape[1]:
            diff_x = subj_conform.CT.shape[1]-xy_original_ct
            left_x = diff_x//2
            right_x = diff_x-left_x
            diff_y = subj_conform.CT.shape[2]-xy_original_ct
            left_y = diff_y//2
            right_y = diff_y-left_y
            crop = tio.transforms.Crop((left_x,right_x,left_y,right_y,0,0))
            subj_conform = crop(subj_conform)
            self.CT_conform_crop = subj_conform.CT
        
        # Resample cropped CT conform to 2mm
        rsl = tio.transforms.Resample(2)
        subj_rsl = rsl(subj_conform)

        # Load NAC
        self.NAC = tio.ScalarImage(self.NAC_path)
        
        # Resample NAC to CT 2mm
        rsl2 = tio.transforms.Resample(subj_rsl.CT)
        NAC_rsl = rsl2(self.NAC)

        # Adjust for weight and dose
        if self.tracer == 'Fluorodeoxyglucose' and self.weight is not None and self.dose is not None:
            const = 3*(self.weight/self.dose)
            if self.verbose:
                print('\tConstant for adjustment: %s' %const)
            NAC_rsl.set_data(NAC_rsl.data*const)
            
        # Padding (if needed)
        if (padding := self.patch_size[0] > NAC_rsl.shape[1]):
            diff_x = self.patch_size[0]-NAC_rsl.shape[1] # 256-249 = 7
            left_x = diff_x//2
            right_x = diff_x-left_x
            diff_y = self.patch_size[1]-NAC_rsl.shape[2] # 256-249 = 7
            left_y = diff_y//2
            right_y = diff_y-left_y
            self.pad = tio.transforms.Pad((left_x,right_x,left_y,right_y,0,0))
            NAC_rsl = self.pad(NAC_rsl)
        self.padding = padding

        # Normalize
        norm_pet_percentile_normalization = tio.Lambda(lambda x: percentile_norm(x))
        self.NAC_preprocessed = norm_pet_percentile_normalization(NAC_rsl)


    def inference(self, bs=1):        
        subject = tio.Subject(img=self.NAC_preprocessed)
        grid_sampler = tio.data.GridSampler(subject, self.patch_size, self.patch_overlap, padding_mode='constant')
        patch_loader = DataLoader(grid_sampler, batch_size=bs)
        aggregator = tio.data.GridAggregator(grid_sampler, overlap_mode='hann')
        
        if self.verbose:
            print('\tStarting inference')
            start_time = time.time()
        
        for patches_batch in tqdm(patch_loader):
            patch_x = patches_batch['img'][tio.DATA].float().numpy()
            locations = patches_batch[tio.LOCATION]
            ort_outs = self.model.run(None, {self.input_name: patch_x})
            patch_y = torch.from_numpy(ort_outs[0])
            aggregator.add_batch(patch_y, locations)

        self.sCT_preproc_space = tio.ScalarImage(tensor=aggregator.get_output_tensor(), affine=self.NAC_preprocessed.affine)
        if self.verbose:
            print(f'\tInference done in {time.time()-start_time:.01f} seconds')
            
            
    def postprocess(self):
        
        if self.verbose:
            print("\tPostprocessing")
        
        subj = tio.Subject(sCT = self.sCT_preproc_space) # Q-Maria: , affine=subject.input0.affine)) ??
        
        # De normalize
        inv_norm_ct_normalization = tio.Lambda(lambda x: x*2000.0 - 1024.0)
        subj_HU = inv_norm_ct_normalization(subj)
    
        # Resampling sCT from 2mmm
        rsl_from_2mm = tio.transforms.Resample(self.CT_conform_crop)
        subj_conform_cropped = rsl_from_2mm(subj_HU)

        # Padding sCT
        original_CT_shape = 799 # Q-Maria: ??
        if subj_conform_cropped.sCT.shape[0] < original_CT_shape:
            diff_x = original_CT_shape-subj_conform_cropped.sCT.shape[1] # 256-249 = 7
            left_x = diff_x//2
            right_x = diff_x-left_x
            diff_y = original_CT_shape-subj_conform_cropped.sCT.shape[2] # 256-249 = 7
            left_y = diff_y//2
            right_y = diff_y-left_y
            pad = tio.Pad((left_x,right_x,left_y,right_y,0,0), padding_mode=-1024)
            subj_padded = pad(subj_conform_cropped)
            
        # Removing conform
        rsl2 = tio.transforms.Resample(self.CT_path)
        subj_final = rsl2(subj_padded) # Q-Maria: OBS - fails when above if is false..

        # Inserting bed
        CT_bed = tio.ScalarImage(self.CT_path)
        (sCT_np, CT_bed_rsl_np, mask_rsl_np) = (subj_final.sCT.data.numpy()[0], CT_bed.data.numpy()[0], self.bed_mask.data.numpy()[0])

        sCT_np[mask_rsl_np > 0] = CT_bed_rsl_np[mask_rsl_np > 0]
        tc_sCT = torch.unsqueeze(torch.from_numpy(sCT_np), 0)
        self.subj_final_wBed = tio.ScalarImage(tensor = tc_sCT, affine = subj_final.sCT.affine)
    
    
    def save_nii(self, output):
        self.subj_final_wBed.save(output)
        
    
    def save_dicom(self, output):        
        if self.verbose:
            print('\tMaking DICOM')
        
        np_nifti = self.subj_final_wBed.numpy()[0]

        # Force values to lie within a range accepted by the dicom container
        np_nifti = np.maximum( np_nifti, -1024 )
        np_nifti = np.minimum( np_nifti, 3071 )
        
        # Apply rescale intercept
        np_nifti += 1024

        # Flip
        #np_nifti = np.flip(np_nifti, axis=2)

        # Rotate
        np_nifti = np.rot90(np_nifti, axes=(0,1), k=1)

        # Currently does not replace any uid or anything..
        to_dcm(
            np_array=np_nifti,
            dicomcontainer=self.sorted_dicts['CT'],
            dicomfolder=output,
            header_str = f'{self.NACCT_version}_{self.cohort}_{self.tracer}'
        )
        
        self.clean()


def run(input, CT, output, model, batch_size=1, dose=None, weight=None, verbose=False):
    inferer = CTlessPET(verbose)
    
    img_type = "nifti" if str(input).endswith(".nii") or str(input).endswith(".nii.gz") else "dicom"
        
    # Get input
    if img_type == 'dicom':
        NAC, CT = inferer.convert_dicom_data(input, CT)
        
    elif CT is not None:
        NAC = input
    else:
        raise ValueError('You gave a nifty (NAC) file as input but forgot to give a CT file as well.')
        
    inferer.setup(model)
    
    inferer.set_dose_and_weight_and_tracer(dose, weight)
        
    inferer.preprocess(NAC, CT)
    
    inferer.inference(batch_size)
    
    inferer.postprocess()
    
    if img_type == 'dicom':
        inferer.save_dicom(output)
    else:
        inferer.save_nii(output)
    

def convert_NAC_to_sCT():
    
    print("\n########################")
    print("If you are using CTlessPET, please cite the following paper:\n")
    print("Montgomery ME, Andersen FL, d’Este SH, Overbeck N, Cramon PK, Law I, Fischer BM, Ladefoged CN. "
        "Attenuation Correction of Long Axial Field-of-View Positron Emission Tomography Using Synthetic "
        "Computed Tomography Derived from the Emission Data: Application to Low-Count Studies and Multiple Tracers. "
        "Diagnostics. 2023; 13(24):3661. https://doi.org/10.3390/diagnostics13243661")
    print("########################\n")
    
    parser = argparse.ArgumentParser(
            description=(
                "Create synthetic CT from NAC-PET data."
            ),
        )

    # Parameters
    parser.add_argument("-i", "--input", help="Input file (nifti) or directory (dicom).", type=str)
    parser.add_argument("--CT", help="CT container file (nifti) or directory (dicom). Can also be included in input for dicom files", type=str)
    parser.add_argument("-o", "--output", help="Input file (nifti) or directory (dicom). Must be the same format as input.", type=str)
    parser.add_argument("-m", "--model", help="Chose a model to use. Will overwrite the choice automatically selected when using dicom data.", choices=['FDG','FDG_Pediatric','H2O'], type=str)
    parser.add_argument("-b", "--batch_size", help="Batch size", type=int, default=1)
    parser.add_argument("-d", "--dose", help="Injected dose", type=float)
    parser.add_argument("-w", "--weight", help="Patient weight", type=float)
    parser.add_argument("-v", "--verbose", help="Add verbosity", action='store_true')
    args = parser.parse_args()
    
    run(
        input = args.input,
        CT = args.CT,
        output = args.output,
        model = args.model,
        batch_size = args.batch_size,
        dose = args.dose,
        weight = args.weight,
        verbose = args.verbose
    )
    
    
if __name__ == "__main__":
    convert_NAC_to_sCT()