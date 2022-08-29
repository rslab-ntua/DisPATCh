from attr import s
import numpy as np
import os
import rasterio
import gc

from utils import reproj_match, write_image

SSM_FLAGS = {
    "nodata": 255,
    "max": 242,
    "min": 241,
    "water": 251,
    "sensitivity": 252,
    "slope": 253
}

def mask_ssm(array, flag_list):
    
    flag_mask = np.zeros_like(array)
    
    # then we will loop through the flags and add the 
    for flag in flag_list:
        # get the mask for this flag
        flag_mask[array == SSM_FLAGS[flag]] = 1

    return flag_mask

def ssm2volumetric(ssm:str, mask:str, clay:str, sand:str, savepath:str, fname:str = None, ext:str = "vol", nodata:float = None)->str:

    if fname is None:
        fname = os.path.splitext(ssm)[0] + "_" + ext + ".tif"
    
    rclay = os.path.splitext(clay)[0] + "_reproj.tif"
    
    reproj_match(clay, ssm, rclay)

    rsand = os.path.splitext(sand)[0] + "_reproj.tif"
    
    reproj_match(sand, ssm, rsand)
    
    SSM = rasterio.open(ssm)
    metadata = SSM.meta.copy()
    MASK = rasterio.open(mask)
    CLAY = rasterio.open(rclay)
    SAND = rasterio.open(rsand)
    
    clay_array = CLAY.read().astype(np.float32)
    sand_array = SAND.read().astype(np.float32)

    theta_res = 0.15 * clay_array*0.01
    theta_sat = 0.489 - 0.126 * sand_array*0.01
    
    del(clay_array, sand_array)
    gc.collect()

    ssm_array = SSM.read().astype(np.float32)
    mask = MASK.read()

    ssm_vol_array = theta_res + (theta_sat - theta_res) * ssm_array * 0.01
    if nodata is None:
        ssm_vol_array[mask == 1] = metadata["nodata"]
    else:
        ssm_vol_array[mask == 1] = nodata
        metadata.update(nodata = nodata)

    metadata.update(dtype = ssm_vol_array.dtype)

    write_image(os.path.join(savepath, fname), ssm_vol_array, metadata)

    return fname