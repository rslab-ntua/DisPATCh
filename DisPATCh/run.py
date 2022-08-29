import os
import fnmatch
import gc

from DisPATCh import *

# Provide datapath & save path
datapath = "D:/Diplomatiki/Data/Spain-Peiramata/5_IMAGES"
SSM_datapath = "C0146802/"
LS_datapath = "Landsat/"

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
            
            for (dirpath_ssm, _, _) in os.walk(os.path.join(datapath, SSM_datapath)):
                for file2 in os.listdir(dirpath_ssm):
                    # Find data
                    os.path.join(dirpath_ssm, file2)
                    if fnmatch.fnmatch(os.path.join(dirpath_ssm, file2), f'*SSM*{DATE}*.nc'):

                        image = rasterio.open("netcdf:" + os.path.join(dirpath_ssm, file2)+ ":ssm")
                        profile = image.meta.copy()
                        profile.update(driver="Gtiff")
                        SSM_PATH = os.path.splitext(file2)[0] + ".tif"
                        with rasterio.open(os.path.join(DATE_RESULTS, f"{SSM_PATH}"), mode = "w", **profile) as dst:
    
                            dst.update_tags(**image.tags())
                            data = image.read()
                            flag_list = ["nodata", "max", "min", "water", "sensitivity", "slope"]
                            mask = mask_ssm(data, flag_list)
                            data = data * float(image.tags(1)["scale_factor"])
                            data[mask == 1] = 255
                            dst.write(data)


                        # Reading soil moisture data
                        # As Rasterio object (check https://rasterio.readthedocs.io/en/latest/quickstart.html)
                        # !! REQUIRES NO MEMORY and its really fast!! To read as NumPy array simply run e.g array = SSM.read()
                        SSM = rasterio.open(os.path.join(DATE_RESULTS, SSM_PATH), mode = "r")

                        # Read SSM as array
                        ssm_array = SSM.read()
                        ssm_mask = mask_ssm(ssm_array, ["nodata", "max", "min", "water", "sensitivity", "slope"])
                        ssm_mask_path = os.path.join(DATE_RESULTS, SSM_PATH.split(".")[0] + "_MASK.tif")
                        # Get important metadata to write image. 
                        metadata = SSM.meta.copy() # Needs to be copied
                        write_image(ssm_mask_path, ssm_mask, metadata)

                        SSM_PATH = ssm2volumetric(
                                        os.path.join(DATE_RESULTS, SSM_PATH),
                                        ssm_mask_path, 
                                        os.path.join(datapath, "Soil/", "STU_EU_T_CLAY_4326.tif"), 
                                        os.path.join(datapath, "Soil/", "STU_EU_T_SAND_4326.tif"),
                                        DATE_RESULTS)
                                        
                        # Reading masks L8
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

                        ################################################################################################

                        # Resample SSM to 30m (match with LST)
                        infile = os.path.join(DATE_RESULTS, SSM_PATH)
                        match = os.path.join(DATE_RESULTS, LST_PATH)
                        out = SSM_PATH.split(".")[0] + "_USMP.tif" # Setting the new filename based on the reference file
                        SSM_USMP_PATH = os.path.join(DATE_RESULTS, out)
                        reproj_match(infile, match, SSM_USMP_PATH)

                        # Now reading SMAP resampled data as rasterio object again
                        SSM_USMP = rasterio.open(SSM_USMP_PATH, mode = "r")

                        ###################################################################################################
                        
                        # STARTING CALCULATE NDVI AND Fvc
                        ###################################################################################################
                        # Read NIR and Red bands as Numpy arrays
                          
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

                        del(FVC_array)
                        del(LST_array)
                        gc.collect()

                        # Calculate Tv
                        TV_PATH = "_".join(os.path.splitext(LST_PATH)[0].split("_")[:-3]) + "_TV.tif"
                        if not os.path.exists(os.path.join(DATE_RESULTS, TV_PATH)):
                            FVC_array = FVC.read()
                            # Then LST
                            LST_array = LST.read()
                            Tv_array = calculateTv_multiprocessing(FVC_array, LST_array, endmembers)
                            Tv_array[(final_mask == 1) | (FVC_array == FVC.meta["nodata"])] = -9999
                            # Write result
                            metadata = RED.meta.copy()
                            metadata.update(nodata = -9999, dtype = Tv_array.dtype)
                            write_image(os.path.join(DATE_RESULTS, TV_PATH), Tv_array, metadata)
                            del(FVC_array)
                            del(LST_array)
                            del(Tv_array)
                            gc.collect()

                        Tv = rasterio.open(os.path.join(DATE_RESULTS, TV_PATH), mode = "r")
                        
                        TS_PATH = "_".join(os.path.splitext(LST_PATH)[0].split("_")[:-3]) + "_TS.tif"
                        if not os.path.exists(os.path.join(DATE_RESULTS, TS_PATH)):
                            FVC_array = FVC.read()
                            LST_array = LST.read()
                            Tv_array =  Tv.read()
                            Ts_array = calculateTs(FVC_array, LST_array, Tv_array)
                            Ts_array[(final_mask == 1) | (FVC_array == FVC.meta["nodata"])] = -9999
                            # Write result
                            metadata = RED.meta.copy()
                            metadata.update(nodata = -9999, dtype = Ts_array.dtype)
                            write_image(os.path.join(DATE_RESULTS, TS_PATH), Ts_array, metadata)

                            del(FVC_array)
                            del(LST_array)
                            del(Tv_array)
                            del(Ts_array)

                            gc.collect()

                        Ts = rasterio.open(os.path.join(DATE_RESULTS, TS_PATH), mode = "r")

                        SEE_PATH = "_".join(os.path.splitext(LST_PATH)[0].split("_")[:-3]) + "_SEE.tif"
                        if not os.path.exists(os.path.join(DATE_RESULTS, SEE_PATH)):
                            Ts_array = Ts.read()
                            LST_array = LST.read()
                            see_array = Soil_Evaporation_Efficiency(Ts_array, LST_array, endmembers)
                            see_array[(final_mask == 1) | (Ts_array == Ts.meta["nodata"])] = -9999
                            # Write result
                            metadata = RED.meta.copy()
                            metadata.update(nodata = -9999, dtype = Ts_array.dtype)
                            write_image(os.path.join(DATE_RESULTS, SEE_PATH), see_array, metadata)
                            
                            del(Ts_array)
                            del(LST_array)
                            del(see_array)

                            gc.collect()

                        SEE = rasterio.open(os.path.join(DATE_RESULTS, SEE_PATH), mode = "r")

                        # Resample SEE to coarse resolution
                        SEE_PATH_LOW = "_".join(os.path.splitext(LST_PATH)[0].split("_")[:-3]) + "_SEE_COARSE.tif"
                        infile = os.path.join(DATE_RESULTS, SEE_PATH)
                        match = os.path.join(DATE_RESULTS, SSM_PATH)
                        out =  os.path.join(DATE_RESULTS, SEE_PATH_LOW)
                        reproj_match(infile, match, out, resampling = Resampling.average)

                        # Resample SEE (average) coarse resolution back to high resolution
                        SEE_PATH_LOW_HIGH = "_".join(os.path.splitext(LST_PATH)[0].split("_")[:-3]) + "_SEE_COARSE_HIGH.tif"
                        infile = os.path.join(DATE_RESULTS, SEE_PATH_LOW)
                        match = os.path.join(DATE_RESULTS, SEE_PATH)
                        out =  os.path.join(DATE_RESULTS, SEE_PATH_LOW_HIGH)
                        reproj_match(infile, match, out)
                        SEE_LOW_HIGH = rasterio.open(os.path.join(DATE_RESULTS, SEE_PATH_LOW_HIGH), mode = "r")

                        # Resample SSM Mask
                        SSM_MASK = rasterio.open(ssm_mask_path, mode = "r")

                        SSM_MASK_USMP_PATH = os.path.join(DATE_RESULTS, SSM_PATH.split(".")[0] + "_MASK_USMP.tif")
                        infile = ssm_mask_path
                        match =os.path.join(DATE_RESULTS, SSM_PATH.split(".")[0] + "_USMP.tif")
                        out =  os.path.join(DATE_RESULTS, SSM_MASK_USMP_PATH)
                        reproj_match(infile, match, out)

                        SSM_MASK_USMP = rasterio.open(os.path.join(DATE_RESULTS, SSM_MASK_USMP_PATH), mode = "r")

                        SMP_PATH = SSM_PATH.split(".")[0] + "_SMP.tif"
                        if not os.path.exists(os.path.join(DATE_RESULTS, SMP_PATH)):
                            SEE_LOW_HIGH_array = SEE_LOW_HIGH.read() # Coarse resolution SEE upsampled to high resolution
                            SSM_USMP_array = SSM_USMP.read()  # Coarse resolution SSM upsampled to high resolution
                            SMP_array = soil_moisture_parameter(SSM_USMP_array, SEE_LOW_HIGH_array)
                            SSM_MASK_USMP_array = SSM_MASK_USMP.read() # SSM mask at high resolution
                            SMP_array[SSM_MASK_USMP_array == 1] = -9999
                            metadata = SSM_USMP.meta.copy()
                            metadata.update(nodata = -9999, dtype = SMP_array.dtype)
                            write_image(os.path.join(DATE_RESULTS, SMP_PATH), SMP_array, metadata)
                            
                            del(SEE_LOW_HIGH_array)
                            del(SSM_USMP_array)
                            del(SMP_array)
                            gc.collect()

                        SMP = rasterio.open(os.path.join(DATE_RESULTS, SMP_PATH), mode = "r")

                        SMH_PATH = SSM_PATH.split(".")[0] + "_SMH.tif"
                        if not os.path.exists(os.path.join(DATE_RESULTS, SMH_PATH)):
                            
                            SEE_LOW_HIGH_array = SEE_LOW_HIGH.read() # Coarse resolution SEE upsampled to high resolution
                            SEE_array = SEE.read()
                            
                            SSM_USMP_array = SSM_USMP.read()  # Coarse resolution SSM upsampled to high resolution
                            SSM_MASK_USMP_array = SSM_MASK_USMP.read() # SSM mask at high resolution
                            
                            SMP_array = SMP.read()


                            disaggregated = Disaggregation(SSM_USMP_array, SEE_LOW_HIGH_array,SEE_array, SMP_array)
                            disaggregated[(SSM_MASK_USMP_array == 1) | (SEE_array == -9999)] = -9999
                            metadata = SSM_USMP.meta.copy()
                            metadata.update(nodata = -9999, dtype = disaggregated.dtype)
                            write_image(os.path.join(DATE_RESULTS, SMH_PATH), disaggregated, metadata)

                            del(SEE_LOW_HIGH_array)
                            del(SSM_USMP_array)
                            del(SMP_array)
                            del(SEE_array)
                            del(SSM_MASK_USMP_array)
                            del(disaggregated)
                            gc.collect()

                            print("OK")