import os

import numpy as np
import pydicom

import torch
from WL import *


path = 'E:/Lucas_up/eric6/tumor_seg/dicom'

filenames = os.listdir(path)

print(len(filenames))
slices_tumor = np.zeros((len(filenames), 512, 512))
slices_liver = np.zeros((len(filenames), 512, 512))
for i, name in enumerate(filenames):
    name = os.path.join(path, name)
    slice = pydicom.dcmread(name)
    slice_liver = WL(slice, 0, 2048)
    slice_tumor = WL(slice, 100, 150)
    slices_liver[i] = slice_liver
    slices_tumor[i] = slice_tumor

slices_liver_tensor = torch.tensor(slices_liver)
slices_liver_tensor = slices_liver_tensor.unsqueeze(1).float()

































slices_tumor_tensor = torch.tensor(slices_tumor)
slices_tumor_tensor = slices_tumor_tensor.unsqueeze(1).float()

