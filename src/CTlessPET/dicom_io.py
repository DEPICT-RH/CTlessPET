import glob, os
from pydicom import dcmread
from pathlib import Path
import datetime
import numpy as np

def __generate_uid_suffix() -> str:
    """ Generate and return a new UID """
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")

def generate_SeriesInstanceUID() -> str:
    """ Generate and return a new SeriesInstanceUID """
    return '1.3.12.2.1107.5.2.38.51014.{}11111.0.0.0'.format(__generate_uid_suffix())

def generate_SOPInstanceUID(i: int) -> str:
    """ Generate and return a new SOPInstanceUID

    Parameters
    ----------
    i : int
        Running number, typically InstanceNumber of the slice
    """
    return '1.3.12.2.1107.5.2.38.51014.{}{}'.format(__generate_uid_suffix(),i)


def to_dcm(np_array,
           dicomcontainer,
           dicomfolder,
           verbose=False,
           modify=False,
           description=None,
           study_id=None,
           patient_id=None,
           header_str=None):

    # CONSTANT(S)
    _CONSTANTS = {'int16': 32767,
                  'uint16': 65535}

    if verbose:
        print("Converting to DICOM")

    if description or study_id:
        modify = True

    # Get information about the dataset from a single file
    ds = dcmread(next(iter(dicomcontainer.values())))
    data_type = ds.pixel_array.dtype.name

    # Check that the correct number of files exists
    totalSlicesInArray = np_array.shape[2]

    assert len(dicomcontainer) == totalSlicesInArray, 'DicomContainer contained {} slices whereas the final sCT array contained {}'.format(len(dicomcontainer), totalSlicesInArray)

    ## Prepare for MODIFY HEADER
    newSIUID = generate_SeriesInstanceUID()

    # Prepare output folder
    if isinstance(dicomfolder, str):
        dicomfolder = Path(dicomfolder)
    dicomfolder.mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(dicomcontainer.values()):
        ds = dcmread(f)

        # Get single slice
        assert ds.pixel_array.shape == (np_array.shape[0],np_array.shape[1]),  ds.pixel_array.shape + (np_array.shape[0],np_array.shape[1])
        data_slice = np_array[:, :, i].astype('double')
        data_slice = data_slice.astype(data_type)

        # Insert pixel-data
        ds.PixelData = data_slice.tostring()

        # Update LargesImagetPixelValue tag pr slice
        if 'LargestImagePixelValue' in ds:
            ds.LargestImagePixelValue = int(np.ceil(data_slice.max()))

        if modify:

            # Set information if given
            if description:
                ds.SeriesDescription = description
            if study_id:
                ds.SeriesNumber = study_id
            if patient_id:
                ds.PatientID = patient_id
                ds.PatientName = patient_id

            # Update SOP - unique per file
            ds.SOPInstanceUID = generate_SOPInstanceUID(i+1)

            # Same for all files
            ds.SeriesInstanceUID = newSIUID

        if header_str is not None:
            # Modify Version Header
            # (0009,0010) LO [SIEMENS CT VA1 DUMMY]                   #  20, 1 PrivateCreator
            # (0018,1020) LO [VG76B]                                  #   6, 1 SoftwareVersions
            ds[0x0018,0x1020].value = header_str # Updates version number
            ds[0x0018,0x0060].value = 120 # Makes KVP 120
            
        fname = f"dicom_{i:04}.dcm"
        ds.save_as(dicomfolder.joinpath(fname))
        
        
def get_sort_files_dict(path):

    files = {'CT': {}, 'PT': {}}

    for p in Path(path).rglob('*'):
        if p.name.startswith('.'):
            continue
        if not p.is_file():
            continue
        try:
            ds = dcmread(str(p))
            files[ds.Modality][ds.SliceLocation] = p
            
        except Exception as e:
            print(f"\tSkipping {p}. Not dicom?. Got error: {e}")
            pass
        
    sorted_files = {}
    for k, d in files.items():
        sorted_files[k] = dict(sorted(d.items()))
        
    return sorted_files
