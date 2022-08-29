import os
import rasterio
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from shapely.geometry import Point, LineString, Polygon
import matplotlib.pyplot as plt
import dask.dataframe as dd
import dask_geopandas
import geopandas as gpd

from landsat import LandsatMasker, LandsatConfidence
from ssm import mask_ssm
from utils import *

def ndvi(red, nir, dtype = np.float32, nodata = -9999):
    """Calculate NDVI.

    Args:
        red (np.array, float, int): Red array or value
        nir (np.array, float, int): NIR array or value

    Returns:
        np.array, float: NDVI array or value
    """
    np.seterr(divide = "ignore")
    np.seterr(invalid = "ignore")

    ndvi = (nir - red) / (nir + red)
    
    # Check for invalid values
    if (type(ndvi) is np.array) or (type(ndvi) is np.ndarray):
        ndvi[ndvi < -1] = nodata
        ndvi[ndvi > 1] = nodata
        ndvi = ndvi.astype(dtype)
    else:
        raise TypeError("Only Numpy Arrays are supported.")
    
    return ndvi
    
def fractional_vegetation_cover(ndvi, vegetation = 0.97, soil = 0.01, dtype = np.float32, nodata = -9999):
    """Calculate fractional vegetation cover.

    Args:
        ndvi (np.array, float): NDVI value
        vegetation (float, optional): Denote pure vegetation value. Defaults to 0.9.
        soil (float, optional): Vegetation-free soil value. Defaults to 0.15.

    Returns:
        np.array, float: Factional vegetation cover
    """
    np.seterr(divide='ignore')
    np.seterr(invalid = "ignore")

    fvc = (ndvi - soil) / (vegetation - soil)
    if (type(fvc) is np.array) or (type(fvc) is np.ndarray):
        fvc[fvc < 0] = nodata
        fvc[fvc > 1] = nodata
        fvc = fvc.astype(dtype)
    else:
        raise TypeError("Only Numpy Arrays are supported.")

    return fvc

def get_endmembers(FVC:np.array, LST:np.array)->dict:
    """Find the 4 temperature endmembers (Veg. Max., Veg. Min., Soil max. and Soil min.) values with the use
    of linear regression models.
    
    Args:
        FVC (np.array): Array with the fractional vegetation cover data
        LST (np.array): Array with the land surface temperature data

    Returns:
        dict: Endmember values
    """
    # Get positions with FVC and LST data
    rows, cols = np.where((FVC[0,:,:]>=0) & (LST[0,:,:]>0))
    pixels_fv = FVC[:, rows, cols].tolist()[0]
    pixels_lst = LST[:, rows, cols].tolist()[0]
    space = pd.DataFrame(list(zip(pixels_fv, pixels_lst)), columns = ['fv', 'LST'])
    min_c = 0
    dry_edge = pd.DataFrame(columns = ['fv', 'LST'])
    wet_edge = pd.DataFrame(columns = ['fv', 'LST'])
    for max_c in np.arange(0, 1, 0.05):
        df = space.loc[(space.fv >= min_c) & (space.fv < max_c)].astype(float)
        if df.empty:
                pass
        else:
            argmax = df.LST.argmax() # Returns max position
            argmin = df.LST.argmin() # Returns min position       
            max = df.iloc[argmax].to_frame().T
            min = df.iloc[argmin].to_frame().T
            dry_edge = pd.concat([dry_edge, max])
            wet_edge = pd.concat([wet_edge, min])
            min_c = max_c
    
    # Getting endmembers for zone D
    argmax = space.LST.argmax() # Returns max position
    argmin = space.LST.argmin()
    fvLST_max = space.iloc[argmax].to_frame().T.values.tolist()[0]
    fvLST_min = space.iloc[argmin].to_frame().T.values.tolist()[0]

    # Linear regression for wet edge
    reg = LinearRegression().fit(wet_edge[['fv']], wet_edge[['LST']])
    coef = reg.coef_[0][0]
    intercept = reg.intercept_[0]
    score = reg.score(wet_edge[['fv']], wet_edge[['LST']])
    wet_parameters= {'intercept': intercept, 'coefficient': coef, 'R2': score}
    
    # Same for dry edge
    reg = LinearRegression().fit(dry_edge[['fv']], dry_edge[['LST']])
    coef = reg.coef_[0][0]
    intercept = reg.intercept_[0]
    score = reg.score(dry_edge[['fv']], dry_edge[['LST']])
    dry_parameters = {'intercept': intercept, 'coefficient': coef, 'R2': score}
    
    #TODO: Remove these
    #################################################################
    # Where fractional vegetation = 1 max vegetation temperature (Tvmax) occurs on the dry edge
    #Tvmax = dry_parameters['intercept'] + dry_parameters['coefficient']

    # Where fractional vegetation = 1 min vegetation temperature (Tvmin) occurs on the wet edge
    #Tvmin = wet_parameters['intercept'] + wet_parameters['coefficient']

    # Where fractional vegetation = 0 max soil temperature (Tsmax) occurs on the dry edge
    #Tsmax = dry_parameters['intercept']

    # Where fractional vegetation = 0 min soil temperature (Tsmax) occurs on the wet edge
    #Tsmin = wet_parameters['intercept']
    #################################################################

    # Dry edge
    intercept = fvLST_max[1] - dry_parameters["coefficient"] * fvLST_max[0]
    
    # Where fractional vegetation = 0 max soil temperature (Tsmax) occurs on the dry edge
    Tsmax = intercept
    # Where fractional vegetation = 1 max vegetation temperature (Tvmax) occurs on the dry edge
    Tvmax = intercept + dry_parameters["coefficient"]
    
    
    # Wet edge
    intercept = fvLST_min[1] - wet_parameters["coefficient"] * fvLST_min[0]
    
    # Where fractional vegetation = 0 min soil temperature (Tsmax) occurs on the wet edge
    Tsmin = intercept
    # Where fractional vegetation = 1 min vegetation temperature (Tvmin) occurs on the wet edge
    Tvmin = intercept + wet_parameters["coefficient"]
    
    if (Tvmax - Tvmin) <= 0.5 * (Tsmax - Tsmin):
        Tvmax = Tvmin + 0.5 * (Tsmax - Tsmin)
        
    endmembers = {"vegetation_max": Tvmax, "vegetation_min": Tvmin, "soil_max":Tsmax, "soil_min": Tsmin, "LST_max": fvLST_max[1], "LST_min": fvLST_min[1]}
    
    return endmembers

def find_triangle_point(endmembers:dict)->tuple:      
    """Calculate the point coordinates of the unknown edge of the zone triangles.

    Args:
        endmembers (dict): Temperature endmembers (from get_endmembers)

    Returns:
        tuple: X and Y coordinates as floats
    """
    # Endmember point coordinates
    a1 = np.array([0, endmembers["soil_max"]])
    a2 = np.array([1, endmembers["vegetation_min"]])
    b1 = np.array([0, endmembers["soil_min"]])
    b2 = np.array([1, endmembers["vegetation_max"]])
    # Line 1 from points 1 (fractional vegetation cover = 0, Soil maximum temperature)
    # and 2 (fractional vegetation cover = 1, Vegetation minimum temperature)
    line_1 = LineString([a1, a2])
    # Line 2 from points 3 (fractional vegetation cover = 0, Soil minimum temperature)
    # and 4 (fractional vegetation cover = 1, Vegetation maximum temperature)
    line_2 = LineString([b1, b2])
    # Intersect to find the triangle's 3rd corner coordinates
    intersection = line_1.intersection(line_2)

    x, y = intersection.xy
    x, y = x[0], y[0] # Cause both being returned as weird arrays, just keep the floats. 

    return (x, y)

def get_zones(endmembers:dict, triangle_edge_coords:tuple)->dict:
    """Calculates the 4 fractional vegetation cover-LST zones.

    Args:
        endmembers (dict): Temperature endmembers (from get_endmembers)
        triangle_edge_coords (tuple): Point coordinates of the unknown edge of the zone triangles (from find_triangle_point)

    Returns:
        dict: Shapely polygon geometries of the 4 zones
    """
    # Create Zone A (Triangle) from point with fractional vegetation cover = 0 and Soil maximum temperature, intersected point coordinates (triangle_edge_coords)
    # and fractional vegetation cover = 0, Soil minimum temperature
    zones = {}
    zones["A"] = Polygon(((0, endmembers["soil_max"]), (triangle_edge_coords[0], triangle_edge_coords[1]), (0, endmembers["soil_min"]), (0, endmembers["soil_max"])))
    # The other zones respecrively
    zones["B"] = Polygon(((0, endmembers["soil_max"]), (triangle_edge_coords[0], triangle_edge_coords[1]), (1, endmembers["vegetation_max"]), (0, endmembers["soil_max"])))
    zones["C"] = Polygon(((0, endmembers["soil_min"]), (triangle_edge_coords[0], triangle_edge_coords[1]), (1, endmembers["vegetation_min"]), (0, endmembers["soil_min"])))
    zones["D"] = Polygon(((1, endmembers["vegetation_min"]), (triangle_edge_coords[0], triangle_edge_coords[1]), (1, endmembers["vegetation_max"]), (1, endmembers["vegetation_min"])))

    return zones

def find_zone_containing_point(point:tuple, zones:dict)->str:
    """Returns the name of the zone for a point by intersection. (TO BE REMOVED)

    Args:
        point (tuple): Point coordinates
        zones (dict): Zones (from get_zones)

    Returns:
        str: None if no intersection is found else the name of the zone
    """
    point = Point((point[0], point[1])) # Convert to shapely geometry
    for name, zone in zones.items():
        if zone.intersection(point):
            return name
    
    return None

def get_Tv(fv:float, lst:float, zone:str, endmembers:dict, nodata:float = -9999)->float:
    """Calculates the vegetation temperature.

    Args:
        fv (float): Fractional vegetation cover
        lst (float): Land surface temperature
        zone (str): Zone that the fv-LST point exists
        endmembers (dict): Temperature endmembers (from get_endmembers)
        nodata (float, optional): No data value. Defaults to -9999.

    Returns:
        float: Vegetation temperature
    """

    if zone is not None:
        if zone == "A":
            Tv = (endmembers["vegetation_max"] + endmembers["vegetation_min"]) / 2.
        elif zone == "B":
            vtemp =  (lst - endmembers["soil_max"] * (1-fv)) / fv
            Tv = (vtemp + endmembers["vegetation_max"]) / 2.
        elif zone == "C":
            vtemp = (lst - endmembers["soil_min"] * (1-fv)) / fv
            Tv = (endmembers["vegetation_min"] + vtemp) / 2.
        elif zone == "D":
            Tv = nodata
    else:
        Tv = nodata
    
    return Tv

def get_Tv_array(fv:np.array, lst:np.array, zone:np.array, endmembers:dict, nodata:float = -9999)->np.array:
    
    Tv = np.full(fv.shape, nodata, dtype = np.float32)
    Tv[zone == 1] = (endmembers["vegetation_max"] + endmembers["vegetation_min"]) / 2. # Zone A
    Tv[zone == 2] = (((lst[zone == 2] - endmembers["soil_max"] * (1-fv[zone == 2])) / fv[zone == 2]) + endmembers["vegetation_max"]) / 2. # Zone B
    Tv[zone == 3] = (((lst[zone == 3] - endmembers["soil_min"] * (1-fv[zone == 3])) / fv[zone ==3]) + endmembers["vegetation_min"]) / 2. # Zone C
    Tv[zone == 4] = 0 # add 0 value to Tv at Zone D to calculate later on with extented version the correct SEE for this zone

    return Tv

def calculateTv_multiprocessing(fv:np.array, lst:np.array, endmembers:dict, nodata = -9999)->np.array:
    ##########################
    # MEMORY HUNGRY!
    ##########################
    triangle_edge_coords = find_triangle_point(endmembers) # Variable triangle_edge_coords[0] contains X coordinate and triangle_edge_coords[1] Y respectively
    # Get the 4 zones
    zones = get_zones(endmembers, triangle_edge_coords)
    # Create an empty dataframe to store the Fv-LST data
    df = pd.DataFrame()
    df["fv"] = pd.Series(fv[0,:,:].flatten())
    df["lst"] = pd.Series(lst[0,:,:].flatten())
    # Create a dask DataFrame
    df = dd.from_pandas(df, npartitions = 10)
    # Move to dask GeoDataFrame
    df["geometry"] = dask_geopandas.points_from_xy(df, "fv", "lst")
    gdf = dask_geopandas.from_dask_dataframe(df, geometry="geometry")
    gdf = gdf.compute()

    # Make geoDataFrame from zone triangles (Polygons)    
    triangles = gpd.GeoDataFrame(columns=["zone", "geometry"], geometry="geometry")
    # Replace with integer values
    replace_zones = {"A": 1, "B": 2, "C": 3, "D":4}
    zone_names = pd.Series([replace_zones[key] for key in zones.keys()])
    geometry = gpd.GeoSeries([value for value in zones.values()])
    triangles["geometry"] = geometry
    triangles["zone"] = zone_names
    # Join spatially the data
    gdf = gpd.sjoin(gdf, triangles, how = "left")
    gdf["zone"] = gdf["zone"].fillna(0) 
    # Return to arrays
    zone_array = gdf["zone"].to_numpy(dtype = "uint8")
    zone_array = np.reshape(zone_array, fv.shape)
    Tv = get_Tv_array(fv, lst, zone_array, endmembers)

    return Tv

def calculateTv(fv:np.array, lst:np.array, endmembers:dict, nodata = -9999):
    
    # Calculate the missing triangle (zone) edge
    triangle_edge_coords = find_triangle_point(endmembers) # triangle_edge_coords[0] contains X coordinate and triangle_edge_coords[1] Y respectively
    # Get the 4 zones
    zones = get_zones(endmembers, triangle_edge_coords)
    
    Tv = np.full(fv.shape, nodata)
    ############################################################
    # REALLY SLOW
    ############################################################
    for i in range(0, fv.shape[1]):
        for j in range(0, fv.shape[2]):
            if fv[0,i,j] != nodata:
                point = (fv[0,i,j], lst[0,i,j]) # tuple
                zone = find_zone_containing_point(point, zones)
                Tv[0,i,j] = get_Tv(fv[0,i,j], lst[0,i,j], zone, endmembers)
        
    print ("Vegetation Temperature is ok!")
    
    return(Tv)

def calculateTs(fv:np.array, lst:np.array, Tv:np.array, nodata = -9999):

    Ts = (lst - fv * Tv) / (1 - fv)
    Ts[Tv == 0] = 0
    
    print ("Soil temperature is ok!")

    return Ts

def Soil_Evaporation_Efficiency(Ts, lst, endmembers, nodata = -9999):

    see = (endmembers["soil_max"] - Ts) / (endmembers["soil_max"] - endmembers["soil_min"])
    
    see[(see < 0) | (see > 1)] = nodata
    
    see[Ts == 0] = (endmembers["LST_max"] - lst[Ts == 0]) / (endmembers["LST_max"] - endmembers["LST_min"])
    
    
    print ("SEE is ok!")
    
    return see

def soil_moisture_parameter(soil_moisture:np.array, see:np.array, nodata:int = -9999)->np.array:
    """A value of SMp is obtained for each Soil moisture pixel.
    SMp was set to the soil moisture at field capacity. In DisPATCh,
    SMp is retrieved at 40-km resolution from SMAP and aggregated MODIS data.
    The soil moisture parameter SMp used to compute *SMmod/*SEE.
    Args:
        soil_moisture (np.array): Soil moisture data at the resampled high resolution.
        see (np.array): SEE data resampled to average coarse resolution and then resampled again to high resolution to match soil moisture data.
        nodata (int, optional): No data value to fill. Defaults to -9999.

    Returns:
        np.array: SMp average (from soil moisture data)at high resolution 
    """
    np.seterr(divide = "ignore")
    np.seterr(invalid = "ignore")

    see[(see < 0) | (see > 1)] = nodata

    # Soil moisture parameter SMp calculator
    pi = np.pi
    acos = np.arccos(1 - 2 * see)
    SMp = pi * soil_moisture / acos

    print ("Soil moisture parameter is ok!")
    
    return SMp

def Disaggregation(soil_moisture:np.array, see_coarse:np.array, see_high:np.array, SMp:np.array):
    
    np.seterr(divide = "ignore")
    np.seterr(invalid = "ignore")
    
    D_SMp_SEE = (SMp / np.pi) * ((2)/np.sqrt(1 - (1 - 2 * see_coarse)**2))

    SMh = soil_moisture + D_SMp_SEE * (see_high - see_coarse)

    return SMh

def get_metadata(metadata:str)->tuple:
    import json
    with open(metadata) as json_file:
        data = json.load(json_file)

    return data

def apply_scaling(image:str, fpath:str, scale:float, offset:float, name:str = None, ext:str = "new", nodata:float = 0)->str:
    
    raster = rasterio.open(image)
    array = raster.read().astype(np.float32)
    if nodata is None:
        array = array * float(scale) + float(offset)
    else:
        array[array != nodata] = array[array != nodata] * float(scale) + float(offset)
    metadata = raster.meta.copy()
    metadata.update(dtype = array.dtype)
    
    if name is None:
        new_name = '.'.join(os.path.basename(image).split('.')[:-1]) + f"{ext}.tif"
    else:
        new_name = name
    
    file = os.path.join(fpath, new_name)
    write_image(file, array, metadata)
    
    return new_name