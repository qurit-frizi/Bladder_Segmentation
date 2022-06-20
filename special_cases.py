import SimpleITK as sitk
import os
from scipy.ndimage import binary_closing, binary_opening, binary_dilation
import numpy as np
from scipy.ndimage.morphology import binary_erosion

'''
Script for dealing with bad bladder predictions. Modify this as needed on 
a case by case basis.
'''

def ax_crop(img, below=0, above=200):
    img[below:above, :, :] = 0
    return img
    
def sag_crop(img, left=50, right=100):
    img[:, :, left:right] = 0
    return img

def cor_crop(img, left=50, right=100):
    img[:, left:right, :] = 0
    return img

def closing(img, iterations=10, structure=np.ones((1, 1, 1))):
    dilated = binary_dilation(img, iterations=iterations, structure=structure)
    eroded = binary_erosion(dilated, iterations=iterations, structure=structure)

    return eroded

def opening(img, iterations=10, structure=np.ones((1, 1, 1))):
    eroded = binary_erosion(img, iterations=iterations, structure=structure)
    dilated = binary_dilation(eroded, iterations=iterations, structure=structure)

    return dilated

center = 'PMBCL'
patient_id = 'PMBCL15-17891_2'

# deal with bladder mask
home_dir1 = "/home/ndubljevic/patient_data/bladder_preds/"
filename = f"{patient_id}_bladder.nii"
filepath = os.path.join(home_dir1, filename)

img = sitk.ReadImage(filepath)
spacing = img.GetSpacing()
img_arr = sitk.GetArrayFromImage(img)

# out = opening(img_arr, iterations=2, structure=np.ones((2, 2, 2)))

img_arr[78:, :, :] = 0
out =img_arr.astype(int)

# deal with og pet image
file_dir = f"/home/ndubljevic/patient_data/{center}/niftii_files/"
filename = f"{patient_id}_pt"
filepath = os.path.join(file_dir, filename)
pet_img = sitk.ReadImage(filepath)
pet_img_arr = sitk.GetArrayFromImage(pet_img)

# save new versions
min_suv = np.min(pet_img_arr)
final = np.where(out==1, min_suv, pet_img_arr)
pet_img = sitk.GetImageFromArray(final)
pet_img.SetSpacing(spacing)

home_dir2 = "/home/ndubljevic/patient_data/bladder_removed/"
sitk.WriteImage(pet_img, os.path.join(home_dir2, f'{patient_id}_pt_nobladder_v2.nii'), imageIO='NiftiImageIO')


