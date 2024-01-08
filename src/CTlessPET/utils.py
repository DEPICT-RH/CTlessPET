import os
import shutil
import zipfile
import numpy as np
from urllib.request import urlopen


folder_with_parameter_files = os.path.join(os.path.expanduser('~'), 'CTlessPET_params')

def get_models():
    return  {
        'Default':
            {
                'Fluorodeoxyglucose': 'FDG.onnx',
                'Oxygen-water': 'H2O.onnx'
            },
        'Pediatric':
            {
                'Fluorodeoxyglucose': None,
                'MFBG': None
            }
        }
    
def get_model_versions():
    return  {
        'Default':
            {
                'Fluorodeoxyglucose': 0.1,
                'Oxygen-water': 0.1
            },
        'Pediatric':
            {
                'Fluorodeoxyglucose': 0.1,
                'MFBG': 0.1
            }
        }
    
def get_model_path(cohort,tracer):
    return os.path.join(folder_with_parameter_files, get_models()[cohort][tracer])

def get_model_version(cohort,tracer):
    return get_model_versions()[cohort][tracer]
    
def maybe_download_weights(cohort,tracer):
    download_weights = False
    model_path = get_model_path(cohort,tracer)
    model_version = get_model_version(cohort,tracer)
    model_version_file = model_path.replace('.onnx', '_version')
    
    if not os.path.isfile(model_path):
        download_weights = True
    else:
        if not os.path.isfile(model_version_file):
            download_weights = True
        else:
            existing_version = float(np.loadtxt(model_version_file, dtype=str))
            if model_version > existing_version:
                download_weights = True

    if download_weights:
        
        if not os.path.exists(folder_with_parameter_files):
            os.makedirs(folder_with_parameter_files)
        
        # Delete current file
        if os.path.isfile(model_path):
            os.remove(model_path)

        url = f"https://zenodo.org/records/10463314/files/{cohort}_{tracer}_{model_version}.onnx?download=1"
        print(f"Downloading {cohort}_{tracer}_{model_version}...")
        data = urlopen(url).read()
        with open(model_path, 'wb') as f:
            f.write(data)
            
        # Downloading version file
        url = f"https://zenodo.org/records/10463314/files/{cohort}_{tracer}_version?download=1"
        data = urlopen(url).read()
        with open(model_version_file, 'wb') as f:
            f.write(data)
    
def download_etc_files():
    out_filename = os.path.join(folder_with_parameter_files, "etc", "AC_CT_Leje_full_mask.nii.gz")
    
    if os.path.isfile(out_filename):
        os.remove(out_filename)
        
    if not os.path.exists(os.path.join(folder_with_parameter_files, "etc")):
        os.makedirs(os.path.join(folder_with_parameter_files,"etc"))
        
    url = "https://zenodo.org/records/10463314/files/AC_CT_Leje_full_mask.nii.gz?download=1"
    print("Downloading etc files...")
    data = urlopen(url).read()
    with open(out_filename, 'wb') as f:
        f.write(data)
    
def get_bed_path():
    if not os.path.exists(bed_path := os.path.join(folder_with_parameter_files, 'etc', 'AC_CT_Leje_full_mask.nii.gz')):
        download_etc_files()
    return bed_path
        
        
def percentile_norm(x):
    p005 = np.percentile(x, 0.5)
    p995 = np.percentile(x, 99.5)
    inp_normalised = (x-p005)/(p995-p005)
    return inp_normalised