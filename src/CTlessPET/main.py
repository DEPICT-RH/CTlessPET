import argparse
from typing import Union
from pathlib import Path, PosixPath
import warnings
import tempfile
import time
import subprocess as subp
from multiprocessing import Process
import torchio as tio
import numpy as np
from torch.utils.data import DataLoader
import torch
from pydicom import dcmread
from CTlessPET.utils import (
    maybe_download_weights,
    get_models,
    get_model_paths,
    get_bed_path,
    get_preprocessing_transform,
    de_normalize
)
from CTlessPET.dicom_io import (
    get_sort_files_dict,
    to_dcm
)
import shutil
import dicom2nifti
from tqdm import tqdm

#suppress warnings
warnings.filterwarnings('ignore')


class CTlessPET():
    def __init__(self, debug=False, verbose=False):
        self.verbose = verbose
        self.debug = debug
        if self.debug is not None:
            self.debug_tmp_dir = Path(self.debug)
            Path(self.debug_tmp_dir).mkdir(exist_ok=True, parents=True)
            if self.verbose:
                print("\t[DEBUG] Allocated tmp folder", self.debug_tmp_dir)
        self.tracer = None
        self.patient_age = None
        
        self.NACCT_version = 'V0.2'
        
        
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
        self.tracer = d['RadiopharmaceuticalInformationSequence'][0]['Radiopharmaceutical'].value
        if self.verbose:
            print(f"\tInferred tracer: {self.tracer}")
        if 'PatientAge' in d:
            if d.PatientAge.endswith('Y'):
                self.patient_age = int(d.PatientAge.replace('Y',''))
            elif d.PatientAge.endswith('M'):
                self.patient_age = int(d.PatientAge.replace('M','')) / 12
            elif d.PatientAge.endswith('D'):
                self.patient_age = int(d.PatientAge.replace('D','')) / 365
            if self.verbose:
                print(f"\tInferred patient age: {self.patient_age:.1f} years")
                
        if self.verbose:
            print("\tConverting files to nifti")
            start_time = time.time()
        path_CT_nii = self.tmp_dir / 'CT.nii.gz'
        path_NAC_nii = self.tmp_dir / 'NAC.nii.gz'
        dicom2nifti.dicom_series_to_nifti(self.path_CT, path_CT_nii, reorient_nifti=True)
        dicom2nifti.dicom_series_to_nifti(self.path_NAC, path_NAC_nii, reorient_nifti=True)
        if self.verbose:
            print(f'\tConverting files to nifti done in {time.time()-start_time:.01f} seconds')
            
        if self.debug:
            shutil.copyfile(path_CT_nii, self.debug_tmp_dir / 'CT.nii.gz')
            shutil.copyfile(path_NAC_nii, self.debug_tmp_dir / 'NAC.nii.gz')

        return path_NAC_nii, path_CT_nii
    

    def clean(self):
        self.tmp_dir_object.cleanup()
        

    # Sets up variables
    def setup(self, model, fast): # Model = FDG, FDG_Pediatric
        
        # Get the type of model from the DICOM data if not already set
        if model is None:
            if self.tracer is None:
                raise ValueError('The choice for a model could not be inferred from the input data. Please specify.')
            else:
                if self.patient_age is not None and self.patient_age < 18:
                    if self.verbose:
                        print(f"\tPatient age {self.patient_age} < 18 years. Using pediatric model for tracer {self.tracer}.")
                    self.cohort = 'Pediatric'
                else:
                    self.cohort = 'Default'
                    
                if self.tracer not in get_models()[self.cohort]:
                    raise ValueError(f'Not implemented for the tracer {self.tracer} yet')

        elif model == 'FDG':
            self.cohort = 'Default'
            self.tracer = 'Fluorodeoxyglucose'
        elif model == 'FDG_Pediatric':
            self.cohort = 'Pediatric'
            self.tracer = 'Fluorodeoxyglucose'
        elif model == 'mFBG_Pediatric':
            raise ValueError(f'Not implemented for the model {model} yet')
            #self.cohort = 'Pediatric'
            #self.tracer = 'MFBG'    
        elif model == 'Cu64DOTATATE':
            raise ValueError(f'Not implemented for the model {model} yet')
            #self.cohort = 'Default'
            #self.tracer = 'DOTATATE'
        else:
            raise ValueError(f'Not implemented for the model {model} yet')
        
        # Set the size of the cropped area
        if self.cohort == 'Default':
            self.crop_size = [350,300]
        elif self.cohort == 'Pediatric':
            self.crop_size = [300,300]

        # Get the model
        maybe_download_weights(self.cohort, self.tracer)
        if self.verbose:
            print("\tLoading model")
        self.models = []
        for model_path in get_model_paths(self.cohort, self.tracer):
            model = torch.jit.load(model_path)
            model.to("cuda")
            model.eval()
            self.models.append(model)
            if fast:
                break # Only use the first model if fast inference is requested
        if self.verbose:
            print(f"\tModel {self.cohort}_{self.tracer} loaded")

        # Patch details
        self.patch_size = [128,128,32]
        self.patch_overlap = (70,70,24)
        self.data_shape_in = [1] + self.patch_size
            
            
    # Get mask of CT bed - requires CT is given
    def set_mask(self):
        pt = self.CT_subj
        mask = tio.LabelMap(get_bed_path())  # Load the bed mask image
        self.bed_mask = tio.Resample(pt.CT)(mask)  # Resample the bed mask to match CT spacing
        
            
    # Preprocessing
    def preprocess(self, NAC, CT, normalization_masking_threshold=50):
        
        self.NAC_path = NAC
        self.CT_path = CT
        
        # Load CT
        self.CT_subj = tio.Subject(CT = tio.ScalarImage(self.CT_path))
        
        # Compute BED mask from CT if set
        self.set_mask()

        # Resample CT to 2mm
        rsl_2mm = tio.Resample(2)
        ct_rsl = rsl_2mm(self.CT_subj.CT)

        # Load NAC and resampled CT into subject
        subj = tio.Subject(
            nac = tio.ScalarImage(self.NAC_path),
            ct = ct_rsl
        )
        
        # Resample NAC to 2mm CT
        rsl_ct = tio.Resample('ct')
        subj_rsl = rsl_ct(subj)
        
        # Crop
        # TODO:
        # - Should be guided by a foreground mask!
        # - 
        crop = tio.CropOrPad((self.crop_size[0],self.crop_size[1],subj_rsl.nac.shape[-1]))
        subj_rsl_crop = crop(subj_rsl)
        
        # Normalize
        norm_pet_percentile_normalization = tio.RescaleIntensity(out_min_max=(0,1), percentiles=(0.5, 99.5), masking_method=lambda x: x > normalization_masking_threshold, include=['nac'])
        subj_rsl_crop_norm = norm_pet_percentile_normalization(subj_rsl_crop)
        
        # Transform
        preproc = get_preprocessing_transform()
        self.NAC_preprocessed = preproc(subj_rsl_crop_norm.nac)
        
        if self.debug:
            self.NAC_preprocessed.save(self.debug_tmp_dir / 'NAC_preprocessed.nii.gz')
            

    def inference(self, bs=1):        
        subject = tio.Subject(img=self.NAC_preprocessed)
        grid_sampler = tio.data.GridSampler(subject, self.patch_size, self.patch_overlap, padding_mode='constant')
        patch_loader = DataLoader(grid_sampler, batch_size=bs)
        
        if self.verbose:
            print(f'\tStarting inference with {len(self.models)} model(s)...')
            start_time = time.time()
            
        sCT_stack = []
        for model_ind, model in enumerate(self.models):
            if self.verbose:
                print(f'\t\tUsing model fold {model_ind}..')
            
            aggregator = tio.data.GridAggregator(grid_sampler, overlap_mode='hann')
            with torch.no_grad():
                for patches_batch in tqdm(patch_loader):
                    patch_x = patches_batch['img'][tio.DATA].to('cuda')
                    locations = patches_batch[tio.LOCATION]
                    patch_y = model(patch_x)
                    aggregator.add_batch(patch_y.float().cpu(), locations)
            sCT_stack.append(aggregator.get_output_tensor().cpu())
        if len(sCT_stack) == 1:
            sCT_tensor = sCT_stack[0]
        else:
            stacked_tensor = torch.stack(sCT_stack, dim=0)
            sCT_tensor = torch.median(stacked_tensor, dim=0).values

        self.sCT_preproc_space = tio.ScalarImage(tensor=sCT_tensor, affine=self.NAC_preprocessed.affine)
        if self.verbose:
            print(f'\tInference done in {time.time()-start_time:.01f} seconds')
        if self.debug:
            self.sCT_preproc_space.save(self.debug_tmp_dir / 'sCT_preproc_space.nii.gz')
            
            
    def postprocess(self, insert_bed=True):
        
        if self.verbose:
            print("\tPostprocessing")
        
        subj = tio.Subject(sCT = self.sCT_preproc_space)
    
        # Resampling sCT to CT spacing
        rsl_from_2mm = tio.transforms.Resample(self.CT_subj.CT)
        subj_rslCT = rsl_from_2mm(subj)
        
        if self.debug:
            subj_rslCT.sCT.save(self.debug_tmp_dir / 'sCT_rslCT.nii.gz')

        # De normalize (back to HU at 120kvp)
        inv_norm_ct_normalization = tio.Lambda(lambda x: de_normalize(x))
        subj_HU = inv_norm_ct_normalization(subj_rslCT)
        
        if self.debug:
            subj_HU.sCT.save(self.debug_tmp_dir / 'sCT_rslCT_HU.nii.gz')

        if insert_bed:
            # Inserting bed
            CT_bed = tio.ScalarImage(self.CT_path)
            (sCT_np, CT_bed_rsl_np, mask_rsl_np) = (subj_HU.sCT.data.numpy()[0], CT_bed.data.numpy()[0], self.bed_mask.data.numpy()[0])
            sCT_np[mask_rsl_np > 0] = CT_bed_rsl_np[mask_rsl_np > 0]
            tc_sCT = torch.unsqueeze(torch.from_numpy(sCT_np), 0)
            self.sCT_final = tio.ScalarImage(tensor = tc_sCT, affine = subj_HU.sCT.affine)
        else:
            self.sCT_final = subj_HU.sCT
        
        if self.debug:
            self.sCT_final.save(self.debug_tmp_dir / 'sCT_final.nii.gz')
    
    
    def save_nii(self, output):
        self.sCT_final.save(output)
        
    
    def save_dicom(self, output):        
        if self.verbose:
            print('\tMaking DICOM')
        
        np_nifti = self.sCT_final.numpy()[0]

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


def run(input, CT, output, model, insert_bed=True, normalization_masking_threshold=50, batch_size=1, fast=False, debug=False, verbose=False):
    inferer = CTlessPET(debug=debug, verbose=verbose)
    
    img_type = "nifti" if str(input).endswith(".nii") or str(input).endswith(".nii.gz") else "dicom"
        
    # Get input
    if img_type == 'dicom':
        NAC, CT = inferer.convert_dicom_data(input, CT)
        
    elif CT is not None:
        NAC = input
    else:
        raise ValueError('You gave a nifty (NAC) file as input but forgot to give a CT file as well.')
        
    inferer.setup(model, fast)
        
    inferer.preprocess(NAC, CT, normalization_masking_threshold=normalization_masking_threshold)
    
    inferer.inference(batch_size)
    
    inferer.postprocess(insert_bed)
    
    if img_type == 'dicom':
        inferer.save_dicom(output)
    else:
        inferer.save_nii(output)

    print('Image saved successfully!')
    

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
    parser.add_argument("-f", "--fast", help="Only infer using a single fold/model. Default is to use all available folds.", action='store_true')
    parser.add_argument("--no_bed", help="Do not insert the CT bed from CT container into the synthetic CT (Default is on).", action='store_false')
    parser.add_argument("--normalization_masking_threshold", default=50, type=int, help="Lower threshold to calculate percentiles on NAC image. Set this to zero if your intensities are low (e.g. low-dose PET or short acquisitions)")
    parser.add_argument("-d", "--debug_dir", help="Debug by saving intermediate results to this directory", type=str, default=None)
    parser.add_argument("-v", "--verbose", help="Add verbosity", action='store_true')
    args = parser.parse_args()
    
    run(
        input = args.input,
        CT = args.CT,
        output = args.output,
        model = args.model,
        insert_bed = not args.no_bed,
        normalization_masking_threshold = args.normalization_masking_threshold,
        batch_size = args.batch_size,
        fast = args.fast,
        debug = args.debug_dir,
        verbose = args.verbose
    )
    
    
if __name__ == "__main__":
    convert_NAC_to_sCT()