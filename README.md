# CTlessPET 

## Install
`pip install CTlessPET`

## Use
CTlessPET only requires an NAC-PET dataset and a CT dataset. The CT is used as the container for the synthetic CT, and can be an empty CT acquired before the patient enters the scanner. The NAC-PET should be reconstructed using OSEM with Time-of-Flight enabled.

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
The network has been trained for FDG-PET (adult and pediatric) as well as H20-PET.

The type is automatically selected when running the model with dicom data. You can overwrite the choice of the model using the `--model` flag, e.g. `--model FDG_Pediatric`.

### Optional arguments
You can change the batch size using `--batch_size` as well as overwrite the dose (`--dose`) and weight (`--weight`)  given to the patient. This is otherwise automatically read from the dicom file (if supplied).



## Citation
If you are using CTlessPET, please cite the following paper:

Montgomery ME, Andersen FL, d’Este SH, Overbeck N, Cramon PK, Law I, Fischer BM, Ladefoged CN. 
Attenuation Correction of Long Axial Field-of-View Positron Emission Tomography Using Synthetic
Computed Tomography Derived from the Emission Data: Application to Low-Count Studies and Multiple Tracers.
Diagnostics. 2023; 13(24):3661. https://doi.org/10.3390/diagnostics13243661