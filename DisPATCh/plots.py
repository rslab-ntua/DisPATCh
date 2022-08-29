import os
import gc
import pandas as pd
import fnmatch
import rasterio
from matplotlib import pyplot as plt

from DisPATCh import *
from landsat import LandsatMasker, LandsatConfidence
from utils import *

def plot(space:pd.DataFrame, endmembers:dict, zones:dict):
    pass


datapath = "D:/Diplomatiki/Data/Spain-Peiramata/5_IMAGES/"
LS_datapath = "Landsat/"
savepath = "Figures/"
if not os.path.exists(os.path.join(datapath, savepath)):
    os.mkdir(os.path.join(datapath, savepath))
    
for (dirpath, _, _) in os.walk(os.path.join(datapath, LS_datapath)):
    for file in os.listdir(dirpath):
        # Find data
        if fnmatch.fnmatch(str(file), 'LC08*T1*MTL.json'):
            metadata = get_metadata(os.path.join(dirpath, file))
            
            DATE =  "".join(metadata["LANDSAT_METADATA_FILE"]["IMAGE_ATTRIBUTES"]["DATE_ACQUIRED"].split("-"))

            RESULTS = os.path.join(datapath, "Results")
            if not os.path.exists(RESULTS):
                os.mkdir(RESULTS)
            
            DATE_RESULTS = os.path.join(RESULTS, DATE)
            if not os.path.exists(DATE_RESULTS):
                os.mkdir(DATE_RESULTS)
                
            PRODUCT_ID = metadata["LANDSAT_METADATA_FILE"]["PRODUCT_CONTENTS"]["LANDSAT_PRODUCT_ID"]
            RED_PATH = metadata["LANDSAT_METADATA_FILE"]["PRODUCT_CONTENTS"]["FILE_NAME_BAND_4"]
            NIR_PATH = metadata["LANDSAT_METADATA_FILE"]["PRODUCT_CONTENTS"]["FILE_NAME_BAND_5"]
            LST_PATH = metadata["LANDSAT_METADATA_FILE"]["PRODUCT_CONTENTS"]["FILE_NAME_BAND_ST_B10"]
            PIXEL_QA = metadata["LANDSAT_METADATA_FILE"]["PRODUCT_CONTENTS"]["FILE_NAME_QUALITY_L1_PIXEL"]
            
            MULTI_SCALE_RED = metadata["LANDSAT_METADATA_FILE"]["LEVEL2_SURFACE_REFLECTANCE_PARAMETERS"]["REFLECTANCE_MULT_BAND_4"]
            MULTI_SCALE_NIR = metadata["LANDSAT_METADATA_FILE"]["LEVEL2_SURFACE_REFLECTANCE_PARAMETERS"]["REFLECTANCE_MULT_BAND_5"]
            LST_SCALE = metadata["LANDSAT_METADATA_FILE"]["LEVEL2_SURFACE_TEMPERATURE_PARAMETERS"]["TEMPERATURE_MULT_BAND_ST_B10"]

            MULTI_OFFSET_RED = metadata["LANDSAT_METADATA_FILE"]["LEVEL2_SURFACE_REFLECTANCE_PARAMETERS"]["REFLECTANCE_ADD_BAND_4"]
            MULTI_OFFSET_NIR = metadata["LANDSAT_METADATA_FILE"]["LEVEL2_SURFACE_REFLECTANCE_PARAMETERS"]["REFLECTANCE_ADD_BAND_5"]            
            LST_OFFSET = metadata["LANDSAT_METADATA_FILE"]["LEVEL2_SURFACE_TEMPERATURE_PARAMETERS"]["TEMPERATURE_ADD_BAND_ST_B10"]
            
            RED_PATH = apply_scaling(os.path.join(dirpath, RED_PATH), DATE_RESULTS, MULTI_SCALE_RED, MULTI_OFFSET_RED, ext = "_SCALE_FACTOR")
            NIR_PATH = apply_scaling(os.path.join(dirpath, NIR_PATH), DATE_RESULTS, MULTI_SCALE_NIR, MULTI_OFFSET_NIR, ext = "_SCALE_FACTOR")
            LST_PATH = apply_scaling(os.path.join(dirpath, LST_PATH), DATE_RESULTS, LST_SCALE, LST_OFFSET, ext = "_SCALE_FACTOR")
            
            # LST
            LST = rasterio.open(os.path.join(DATE_RESULTS, LST_PATH), mode = "r")
            # RED
            RED = rasterio.open(os.path.join(DATE_RESULTS, RED_PATH), mode = "r")
            # NIR
            NIR = rasterio.open(os.path.join(DATE_RESULTS, NIR_PATH), mode = "r")

            # Reading masks

            # First for L8
            masker = LandsatMasker(os.path.join(datapath, LS_datapath, PRODUCT_ID, PIXEL_QA), collection = 2)
            conf = LandsatConfidence.none # Remove any pixel stored as cloud

            cloud_name = os.path.join(DATE_RESULTS, "_".join(os.path.splitext(RED_PATH)[0].split("_")[:-3]) + "_CLOUD.tif")
            mask = masker.get_cloud_mask(conf)
            masker.save_tif(mask, cloud_name)

            cloud_shadow_name = os.path.join(DATE_RESULTS, "_".join(os.path.splitext(RED_PATH)[0].split("_")[:-3]) + "_CLOUD_SHADOW.tif")
            mask2 = masker.get_cloud_shadow_mask(conf)
            masker.save_tif(mask2, cloud_shadow_name)

            water_name = os.path.join(DATE_RESULTS, "_".join(os.path.splitext(RED_PATH)[0].split("_")[:-3]) + "_WATER.tif")
            mask3 = masker.get_water_mask(conf)
            masker.save_tif(mask3, water_name)

            # Save all masks as one
            final_mask_name = os.path.join(DATE_RESULTS, "_".join(os.path.splitext(RED_PATH)[0].split("_")[:-3]) + "_MASKS.tif")
            final_mask = mask | mask2 | mask3
            masker.save_tif(final_mask, final_mask_name)


            NDVI_PATH = "_".join(os.path.splitext(RED_PATH)[0].split("_")[:-3]) + "_NDVI.tif"
            # If NDVI image exists don't do the same stuff
            if not os.path.exists(os.path.join(DATE_RESULTS, NDVI_PATH)):
                RED_array = RED.read()
                NIR_array = NIR.read()
                # Compute NDVI
                NDVI_array = ndvi(RED_array, NIR_array)
                NDVI_array[final_mask == 1] = -9999
                # Get important metadata to write image. 
                metadata = RED.meta.copy() # Needs to be copied. Else points in memory to RED metadata object
                # Update no data value to value set in the calculating function (default is -9999)
                metadata.update(nodata = -9999, dtype = NDVI_array.dtype)
                # Write NDVI to disk with the same
                write_image(os.path.join(DATE_RESULTS, NDVI_PATH), NDVI_array, metadata)
                del(NDVI_array)
                del(RED_array)
                del(NIR_array)
                gc.collect()

            NDVI = rasterio.open(os.path.join(DATE_RESULTS, NDVI_PATH), mode = "r")

            FVC_PATH = "_".join(os.path.splitext(RED_PATH)[0].split("_")[:-3]) + "_FVC.tif"
            if not os.path.exists(os.path.join(DATE_RESULTS, FVC_PATH)):
                NDVI_array = NDVI.read()      
                # Compute Fvc
                FVC_array = fractional_vegetation_cover(NDVI_array, dtype = NDVI_array.dtype)
                FVC_array[final_mask == 1] = -9999
                # Get important metadata to write image. 
                metadata = RED.meta.copy() # Needs to be copied. Else points in memory to RED metadata object
                # Update no data value to value setted in the calculating function (default is -9999)
                metadata.update(nodata = -9999, dtype = FVC_array.dtype)
                # Write Fvc to disk with the same
                write_image(os.path.join(DATE_RESULTS, FVC_PATH), FVC_array, metadata)
                del(FVC_array)
                del(NDVI_array)
                gc.collect()

            FVC = rasterio.open(os.path.join(DATE_RESULTS, FVC_PATH), mode = "r")

            FVC_array = FVC.read()
            LST_array = LST.read()
            endmembers = get_endmembers(FVC_array, LST_array)

            
            endmembers = get_endmembers(FVC_array, LST_array)
            triangle_edge_coords = find_triangle_point(endmembers) # Variable triangle_edge_coords[0] contains X coordinate and triangle_edge_coords[1] Y respectively
            # Get the 4 zones
            # Create an empty dataframe to store the Fv-LST data
            df = pd.DataFrame()
            df["fv"] = pd.Series(FVC_array[0,:,:].flatten())
            df["lst"] = pd.Series(LST_array[0,:,:].flatten())
            df = df[(df["fv"] != -9999) & (df["lst"] != 0)]
            # Wet line
            wcl = 'b'
            wwl = 1.4
            
            # Dry line
            dcl = 'r'
            dwl = 1.4
            
            # Labels
            x_label = r'$Fractional\ green\ vegetation\ cover\ (f_{v})$'
            y_label = r'$Temperature\ (^{o})$'
            
            ls = 9
            
            # Title
            title = None
                
            ts = 11

            # Data scatter
            acs = 'black'
            
            amarker = 'x'
            # All scatter size
            ass = 10.
            print("I AM HERE!!!!")
            plt.scatter(df[['fv']], df[['lst']], s = ass, c = acs, marker = amarker) 
            plt.plot([0, 1], [endmembers["soil_max"], endmembers["vegetation_max"]], c = dcl, linewidth = dwl, label = 'Dry Edge')
            plt.plot([0, 1], [endmembers["soil_min"], endmembers["vegetation_min"]], c = wcl, linewidth = wwl, label = 'Wet Edge')
            plt.plot([0, 1], [endmembers["soil_max"], endmembers["vegetation_min"]], c = "yellow", linewidth = 1.4)
            plt.plot([0 ,1], [endmembers["soil_min"], endmembers["vegetation_max"]], c = "yellow", linewidth = 1.4)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.savefig(os.path.join(datapath, savepath, f"fv_lst-{DATE}.png"))
            plt.close()