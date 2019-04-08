import numpy as np
import pydicom

def WL(data, WC, WW): 
    # WC: 窗位     WW：窗宽
    data = data.pixel_array*data.RescaleSlope + data.RescaleIntercept
    min = (2*WC - WW) / 2.0
    max = (2*WC + WW) / 2.0
    # print(max, min)
    idx_max = np.where(data > max)
    idx_min = np.where(data < min)
    idx_in = np.where((data >= min) & (data <= max))
    
    data = (data -min) * 254 / (max - min)
    data[idx_max] = 255
    data[idx_min] = 0
    return data

def WL_NII(data, WC, WW): 
    # WC: 窗位     WW：窗宽
    data = data.pixel_array*data.RescaleSlope + data.RescaleIntercept
    min = (2*WC - WW) / 2.0
    max = (2*WC + WW) / 2.0
    # print(max, min)
    idx_max = np.where(data > max)
    idx_min = np.where(data < min)
    idx_in = np.where((data >= min) & (data <= max))
    
    data = (data -min) * 254 / (max - min)
    data[idx_max] = 255
    data[idx_min] = 0
    return data