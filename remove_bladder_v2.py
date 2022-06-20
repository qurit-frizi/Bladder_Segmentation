import os
import sys
import matplotlib
import cc3d
from skimage.transform import resize

import numpy as np
import yaml
import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt
from skimage import segmentation
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
import torch
from torch.utils.data import DataLoader, Dataset

from unet_code.unet import build_model


'''
This script will generate bladder predictions on PET images. It will save
the bladder mask in PET dimensions in a folder of your choosing, as well
as the original PET image with the bladder area removed.

There is also a check in place that will warn you if the prediction has more
than one disconnected region.
'''

cfg = { 'MODEL': { 'BACKBONE': {'NAME': 'UNet3D'}, 
        'META_ARCHITECTURE': 'SemanticSegNet', 
        'UNET': {'f_maps': 32, 'in_channels': 1, 'out_channels': 3, } } 
        }

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = build_model(cfg, device)
pretrained_file = 'Running_Model/model_best.pth'
model.load_state_dict(torch.load(pretrained_file, map_location=torch.device(device))['model'])
model.eval()
model.double()

class TestDataset(Dataset):
    """Dataset to run bladder removal predictions on"""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with paths to the test images
            root_dir (string): Overarching directory to patient data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ids = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        row = self.ids.iloc[idx, :]
        sub_path = row['sub_path']
        filename = row['filename']
        img_path = row['img_path']
        img = sitk.ReadImage(img_path)
        # Model trained on frames size 128x128, representing a 67% centre crop of original full size (192x192) PET scan 

        # if it's not 192 x 192 x depth, resample to that
        img_arr = sitk.GetArrayFromImage(img) # now has dims of z, y, x
        slices = img_arr.shape[0]
        img_arr = resize(img_arr, (slices, 192, 192), order=1, mode='edge', preserve_range=True)

        if slices > 350:  # if it's very deep
            img_arr = img_arr[215:, :, :]
            slices = img_arr.shape[0]

        target_depth = int(0.4 * slices)  # 0.4 assumes around 250 slices
        img_arr = img_arr[0:target_depth, 32:160, 32:160]  # crop to ~100 x 128 x 128
        img_arr = np.expand_dims(img_arr, axis=[0])  # wants dims of (1, 1, depth, 128, 128)

        torch_img = torch.DoubleTensor(img_arr)
        return torch_img

home_dir = '/home/ndubljevic/patient_data/'
bladder_mask_dir = '/home/ndubljevic/patient_data/bladder_masks/'
bladder_removed_dir = '/home/ndubljevic/patient_data/bladder_masks/'

test_dataset = TestDataset('Running_Model/bladder_removal_ids.csv', home_dir)
test_dataloader = DataLoader(test_dataset, batch_size=1)

ids = pd.read_csv('Running_Model/bladder_removal_ids.csv')

for sample_idx, test_sample in enumerate(test_dataloader):
    preds = model(test_sample).detach()

    ### Do whatever you'd like with predictions ###
    preds = torch.squeeze(preds)  # 1st dim is for classes. 0 is background, 1 bladder, 2 lesions
    post_pred = preds.cpu().softmax(dim=0).numpy()
    pred = np.where(post_pred[1, :, :, :] > 0.5, 1, 0)
    pet_arr = torch.squeeze(test_sample).cpu().numpy()

    # returns mask and number of disconnected regions
    mask_out, N = cc3d.connected_components(pred, connectivity=26, return_N=True)

    # use sample idx to find path of original PET image from CSV file
    row = ids.iloc[sample_idx, :]
    img_path = row['img_path']
    patient_id = row['patient_id']

    # If there is more than 1 disconnected component, then the prediction is probably bad.
    if N > 1:  # 
        print(f'Patient {patient_id} has {N} regions. Look at it more closely!')

    # load that PET image
    og_pet_img = sitk.ReadImage(img_path)
    spacing = og_pet_img.GetSpacing()

    og_pet_arr = sitk.GetArrayFromImage(og_pet_img)
    slices = og_pet_arr.shape[0]

    # figure out how much would have been cropped off
    # note that it's 32 around edges

    if slices > 350:  # if it's pretty deep
        slices_to_bottom = 215
        mod_slices = slices - 215
        target_depth = int(0.4 * mod_slices)
        slices_to_top = int(mod_slices - target_depth)

    else:
        target_depth = int(0.4 * slices)
        slices_to_top = int(slices - target_depth)
        slices_to_bottom = 0

    # add that much of a blank array to the prediction, resize to og dimensions
    pad_pred = np.pad(pred, ((slices_to_bottom, slices_to_top), (32, 32), (32, 32)), constant_values=0)
    pad_pred = resize(pad_pred, og_pet_arr.shape, order=1, mode='edge', preserve_range=True)
    pad_pred = np.round(pad_pred).astype(int)

    # set that region in body to 0 SUV and save it
    min_suv = np.min(og_pet_arr)
    pet_no_blad = np.where(pad_pred==1, min_suv, og_pet_arr)
    pet_img = sitk.GetImageFromArray(pet_no_blad)
    pet_img.SetSpacing(spacing)

    # save PET image with bladder removed
    sitk.WriteImage(pet_img, os.path.join(bladder_removed_dir, f'{patient_id}_pt_nobladder.nii.gz'))

    bladder_img = sitk.GetImageFromArray(pad_pred)
    bladder_img.SetSpacing(spacing)
    # save bladder mask (has dimensions/spacing of PET image)
    sitk.WriteImage(bladder_img, os.path.join(bladder_mask_dir, f'{patient_id}_bladder_mask.nii.gz'))
    print(f'Done patient {patient_id}')
