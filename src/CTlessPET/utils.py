import os
import numpy as np
from urllib.request import urlopen
import torchio as tio
import torch

folder_with_parameter_files = os.path.join(os.path.expanduser('~'), 'CTlessPET_params')

def get_models():
    return  {
        'Default':
            {
                'Fluorodeoxyglucose': (['FDG_k0.pt','FDG_k1.pt','FDG_k2.pt'], 'FDG_version'),
                'DOTATATE': None
            },
        'Pediatric':
            {
                'Fluorodeoxyglucose': (['FDG_Pediatric.pt'], 'FDG_Pediatric_version'),
                'MFBG': None
            }
        }
    
def get_model_versions():
    return  {
        'Default':
            {
                'Fluorodeoxyglucose': 0.2,
                'DOTATATE': None,
                'Oxygen-water': None,
            },
        'Pediatric':
            {
                'Fluorodeoxyglucose': 0.2,
                'MFBG': None
            }
        }
    
def get_model_paths(cohort,tracer):
    return [os.path.join(folder_with_parameter_files, m) for m in get_models()[cohort][tracer][0]]

def get_model_version(cohort,tracer):
    return get_model_versions()[cohort][tracer]

def clean_legacy_weights():
    old_weights = ['FDG.onnx','FDG_Pediatric.onnx']
    for w in old_weights:
        w_path = os.path.join(folder_with_parameter_files, w)
        if os.path.isfile(w_path):
            os.remove(w_path)
    
def maybe_download_weights(cohort,tracer):
    download_weights = False
    model_paths = get_model_paths(cohort,tracer)
    model_version = get_model_version(cohort,tracer)
    model_version_file = os.path.join(folder_with_parameter_files, get_models()[cohort][tracer][1])
    
    download_version_file = True

    for k, model_path in enumerate(model_paths):
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

            url = f"https://zenodo.org/records/17484000/files/{cohort}_{tracer}_k{k}_{model_version}.pt?download=1"
            print(f"Downloading {cohort}_{tracer}_{model_version}...")
            data = urlopen(url).read()
            with open(model_path, 'wb') as f:
                f.write(data)
                
            if download_version_file:
                # Downloading version file
                url = f"https://zenodo.org/records/17484000/files/{cohort}_{tracer}_version?download=1"
                data = urlopen(url).read()
                with open(model_version_file, 'wb') as f:
                    f.write(data)
                download_version_file = False
    
    # Clean up legacy weights
    clean_legacy_weights()
    
def download_etc_files():
    out_filename = os.path.join(folder_with_parameter_files, "etc", "AC_CT_Leje_full_mask.nii.gz")
    
    if os.path.isfile(out_filename):
        os.remove(out_filename)
        
    if not os.path.exists(os.path.join(folder_with_parameter_files, "etc")):
        os.makedirs(os.path.join(folder_with_parameter_files,"etc"))
        
    url = "https://zenodo.org/records/17484000/files/AC_CT_Leje_full_mask.nii.gz?download=1"
    print("Downloading etc files...")
    data = urlopen(url).read()
    with open(out_filename, 'wb') as f:
        f.write(data)
    
def get_bed_path():
    if not os.path.exists(bed_path := os.path.join(folder_with_parameter_files, 'etc', 'AC_CT_Leje_full_mask.nii.gz')):
        download_etc_files()
    return bed_path

def get_preprocessing_transform():
    preprocess = tio.Compose(
        [
            tio.ToCanonical()
        ]
    )
    return preprocess

def de_normalize(normalized_image):

    hu_to_norm = [
        [-1024, -600, 0.0, 0.1],
        [-600, -100, 0.1, 0.3],
        [-100, 100, 0.3, 0.7],
        [100, 300, 0.7, 0.9],
        [300, 1000, 0.9, 1.0]
    ]
    
    img = normalized_image
    device = img.device
    dtype = img.dtype
    hu_img = torch.full_like(img, -1024.0, device=device, dtype=dtype)

    for a, b, c, d in hu_to_norm:
        a_t = torch.tensor(a, device=device, dtype=dtype)
        b_t = torch.tensor(b, device=device, dtype=dtype)
        c_t = torch.tensor(c, device=device, dtype=dtype)
        d_t = torch.tensor(d, device=device, dtype=dtype)

        denorm = ((img - c_t) * (b_t - a_t)) / (d_t - c_t) + a_t
        if c == 0.0:
            mask = (img <= d_t)
        elif d == 1.0:
            mask = (img >= c_t)
        else:
            mask = (img >= c_t) & (img <= d_t)
        hu_img = torch.where(mask, denorm, hu_img)

    return hu_img