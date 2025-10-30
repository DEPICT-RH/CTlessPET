# CTlessPET 

## Install
`pip install CTlessPET`

## Use
CTlessPET only requires an NAC-PET dataset and a CT dataset. The CT is used as the container for the synthetic CT, and can be an empty CT acquired before the patient enters the scanner. The NAC-PET should be reconstructed using OSEM with Time-of-Flight enabled but no PSF modeling. Reconstruction at should be at 440x440 matrix size with a 4 mm Gaussian post filter. Only support for Siemens Bigraph Vision scanners (including the Quadra) has been tested.

### Dicom data
Using a folder containing both NAC and CT data:
```
CTlessPET -i <input_folder> --output <output_folder>
```
or in seperate folders:
```
CTlessPET -i <input_NAC_folder> --CT <input_CT_folder> --output <output_folder>
```

### Nifti data
```
CTlessPET -i <input_NAC_nii> --CT <input_CT_nii> --output <output_nii>
```

### Choice of model
The network has been trained for FDG-PET (adult and pediatric).

The type is automatically selected when running the model with dicom data. You can overwrite the choice of the model using the `--model` flag, e.g. `--model FDG_Pediatric`.

### Optional arguments
You can change the batch size using `--batch_size`.

Some models (e.g. the adult FDG) comes with multiple kfold submodels. Default is to use the median of the individual predictions. If you only wish to infer using one model to speed the inference up, use `--fast`.

The default is to insert the bed from the empty CT scan used as the container. However, if this is not wanted, you can turn it off with `--no_bed`.

## Citation

![image](https://github.com/DEPICT-RH/CTlessPET/assets/108402980/1de108d4-0d1b-40cb-b88c-ed5d18e5b0c9)

If you are using CTlessPET, please cite the following [manuscript for CTlessPET in adults](https://doi.org/10.3390/diagnostics13243661):

Montgomery ME, Andersen FL, d’Este SH, Overbeck N, Cramon PK, Law I, Fischer BM, Ladefoged CN. 
Attenuation Correction of Long Axial Field-of-View Positron Emission Tomography Using Synthetic
Computed Tomography Derived from the Emission Data: Application to Low-Count Studies and Multiple Tracers.
Diagnostics. 2023; 13(24):3661. https://doi.org/10.3390/diagnostics13243661

or the [pediatric version](https://doi.org/10.3390/diagnostics14242788):

Montgomery ME, Andersen FL, Mathiasen R, Borgwardt L, Andersen KF, Ladefoged CN. 
CT-Free Attenuation Correction in Paediatric Long Axial Field-of-View 
Positron Emission Tomography Using Synthetic CT from Emission Data. 
Diagnostics. 2024; 14(24):2788. https://doi.org/10.3390/diagnostics14242788