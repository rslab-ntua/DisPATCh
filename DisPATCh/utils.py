import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling

def write_image(filename:str, array:np.array, metadata:dict) -> None:
    """Writes an array to a geospatial image.

    Args:
        filename (str): Path and name of the new file
        array (np.array): Numpy array or ndarray to write
        metadata (dict): Metadata of the file as dict. Must have all the required attributes.
    """
    with rasterio.open(filename, "w", **metadata) as file:
        file.write(array)

def reproj_match(image:str, base:str, outfile:str, resampling:rasterio.warp.Resampling = Resampling.nearest) -> None:
    """Reprojects/Resamples an image to a base image.
    Args:
        image (str): Path to input file to reproject/resample
        base (str): Path to raster with desired shape and projection 
        outfile (str): Path to saving Geotiff
    """
    # open input
    with rasterio.open(image) as src:
        # open input to match
        with rasterio.open(base) as match:
            dst_crs = match.crs
            dst_transform = match.meta["transform"]
            dst_width = match.width
            dst_height = match.height
        # set properties for output
        metadata = src.meta.copy()
        metadata.update({"crs": dst_crs,
                           "transform": dst_transform,
                           "width": dst_width,
                           "height": dst_height,
                           })
        with rasterio.open(outfile, "w", **metadata) as dst:
            # iterate through bands and write using reproject function
            for i in range(1, src.count + 1):
                reproject(
                    source = rasterio.band(src, i),
                    destination = rasterio.band(dst, i),
                    src_transform = src.transform,
                    src_crs = src.crs,
                    dst_transform = dst_transform,
                    dst_crs = dst_crs,
                    resampling = resampling)

