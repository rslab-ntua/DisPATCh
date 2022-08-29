import os
import rasterio
import numpy as np

class Masker(object):
    """Provides access to functions that produces masks from remote sensing image, according to its bit structure."""

    def __init__(self, band, **var):

        if type(band) is str:
            if 'band_num' in var:
                self.load_file(band, var['band_num'])
            else:
                self.load_file(band)
        else:
            self.load_data(band)

    def load_file(self, file_path, band_num = 0):
        """Load the QA file from a give path.

        Args:
            file_path (str): Path of band file
            band_num (int, optional): Number of band. Defaults to 0.
        """

        self.file_path = file_path
        extension = os.path.splitext(file_path)[1].lower()

        # load file according to the file format.
        bandfile = rasterio.open(file_path)
        self.band_data = bandfile.read()

    def load_data(self, array):
        """Load the BQA band from a np.array.

        Args:
            array (np.array): Numpy array that contains the band data
        """
        
        self.file_path = None
        self.band_data = array

    def get_mask(self, bit_pos, bit_len, value):
        """Generates mask with given bit information.

        Args:
            bit_pos (int): Position of the specific QA bits in the value string
            bit_len (int):Length of the specific QA bits
            value (int): A value indicating the desired condition

        Returns:
            np.array: Mask array
        """

        bitlen = int('1' * bit_len, 2)

        if type(value) == str:
            value = int(value, 2)

        pos_value = bitlen << bit_pos
        con_value = value << bit_pos
        mask = (self.band_data & pos_value) == con_value

        return mask.astype(int)

    def save_tif(self, mask, file_path):
        """Save the given mask as a Geotiff file.

        Args:
            mask (np.array): Mask array
            file_path (str): Path to store image
        """

        if self.file_path is not None:
            bandfile = rasterio.open(self.file_path)
            metadata = bandfile.meta.copy()
            
            with rasterio.open(file_path, "w", **metadata) as file:
                file.write(mask)

class LandsatConfidence(object):
    """Level of confidence that a condition exists.
    high        -	Algorithm has high confidence that this condition exists (67-100 percent confidence).
    medium 		-	Algorithm has medium confidence that this condition exists (34-66 percent confidence).
    low 		-	Algorithm has low to no confidence that this condition exists (0-33 percent confidence)
    undefined	- 	Algorithm did not determine the status of this condition.
    none		-	Nothing.
    """
    high = 3
    medium = 2
    low = 1
    undefined = 0
    none = -1

class LandsatMasker(Masker):
    """Provides access to functions that produces masks from quality assessment band of Landsat 8."""

    def __init__(self, file_path, **var):

        if 'collection' not in var:
            raise Exception('Collection number is required for landsat masker.')
        elif var['collection'] != 1 and var['collection'] != 0 and var['collection'] != 2:
            raise Exception('Collection number must be 0, 1 or 2.')

        self.collection = var['collection']
        super(LandsatMasker, self).__init__(file_path)

    def get_no_cloud_mask(self):
        """Generate a mask for pixels with no cloud.

        Returns:
            np.array: A two-dimensional binary mask
        """

        if self.collection == 0:
            raise Exception('Non-cloud mask is not available for pre-collection data.')
        elif self.collection == 1:
            return self.__get_mask(4, 1, 0, False).astype(int)
        elif self.collection == 2:
            return self.__get_mask(3, 1, 0, False).astype(int)
    
    def get_cloud_shadow_mask(self, conf, cumulative = False):
        """Generate a cloud shadow mask. Note that the cloud shadow mask is only available for collection 1 & 2 data.

        Args:
            conf (int): Level of confidence that water body exists.
            cumulative (bool, optional): A boolean value indicating whether to get masker with confidence value larger than the given one. Defaults to False.

        Returns:
            _type_: A two-dimension binary mask.
        """

        if self.collection == 0:
            raise Exception('Cloud shadow mask is not available in collection 0.')
        elif self.collection == 1:
            return self.__get_mask(7, 3, conf, cumulative).astype(int)
        
        elif self.collection == 2 and (conf is None or conf == -1):
            return self.__get_mask(4, 1, 1, False).astype(int)
        elif self.collection == 2:
            return self.__get_mask(10, 3, conf, cumulative).astype(int)

    def get_cirrus_mask(self, conf, cumulative = False):
        """Generate a cirrus mask. Note that the cirrus mask is only available for Landsat-8 images.

        Args:
            conf (int): Level of confidence that cloud exists
            cumulative (bool, optional): A boolean value indicating whether to get masker with confidence value larger than the given one. Defaults to False.

        Returns:
            np.array: A two-dimension binary mask
        """

        if self.collection == 0:
            return self.__get_mask(12, 3, conf, cumulative).astype(int)
        elif self.collection == 1:
            return self.__get_mask(11, 3, conf, cumulative).astype(int)
        
        elif self.collection == 2 and (conf is None or conf == -1):
            return self.__get_mask(2, 1, 1, False).astype(int)
        elif self.collection == 2:
            return self.__get_mask(14, 3, conf, cumulative).astype(int)

    def get_water_mask(self, conf, cumulative = False):
        """Generate a water body mask. NEEDS MORE TESTING.

        Args:
            conf (int): Level of confidence that water body exists.
            cumulative (bool, optional): A boolean value indicating whether to get masker with confidence value larger than the given one. Defaults to False.

        Returns:
            np.array: A two-dimension binary mask.
        """

        if self.collection == 0:
            if conf == 1 or conf == 3:
                raise Exception('Creating water mask for pre-collection data only accepts confidence value 0 and 2.')

            return self.__get_mask(4, 3, conf, cumulative).astype(int)
        elif self.collection == 1:
            raise Exception('Water mask is not available for Lansat pre-collection data.')
        elif self.collection == 2:
            return self.__get_mask(7, 1, 1, False).astype(int)

    def get_snow_mask(self, conf, cumulative = False):
        """Generate a snow/ice mask.

        Args:
            conf (int): Level of confidence that snow/ice exists.
            cumulative (bool, optional): A boolean value indicating whether to get masker with confidence value larger than the given one. Defaults to False.

        Returns:
            np.array: A two-dimension binary mask.
        """

        if self.collection == 0:
            if conf == 1 or conf == 2:
                raise Exception('Creating snow mask for pre-collection data only accepts confidence value 0 and 3.')

            return self.__get_mask(10, 3, conf, cumulative).astype(int)
        elif self.collection == 1:
            return self.__get_mask(9, 3, conf, cumulative).astype(int)
        
        elif self.collection == 2 and (conf is None or conf == -1):
            return self.__get_mask(5, 1, 1, False).astype(int)
        elif self.collection == 2:
            return self.__get_mask(12, 3, conf, cumulative).astype(int)
    
    def get_fill_mask(self):
        """Generate a mask for designated fill pixels.

        Returns:
            np.array: A two-dimension binary mask.
        """

        return self.__get_mask(0, 1, 1, False).astype(int)
    
    def get_cloud_mask(self, conf=None, cumulative=False):
        """Generate a cloud mask.

        Args:
            conf (int, optional): Level of confidence that cloud exists. Defaults to None.
            cumulative (bool, optional): A boolean value indicating whether to get masker with confidence value larger than the given one. Defaults to False.

        Returns:
            np.array: A two-dimension binary mask
        """

        if self.collection == 0:
            if conf is None or conf == -1:
                raise Exception('Confidence value is required for creating cloud mask from pre-collection data.')
            return self.__get_mask(14, 3, conf, cumulative).astype(int)

        elif self.collection == 1 and (conf is None or conf == -1):
            return self.__get_mask(4, 1, 1, False).astype(int)
        elif self.collection == 1:
            return self.__get_mask(5, 3, conf, cumulative).astype(int)
        
        elif self.collection == 2 and (conf is None or conf == -1):
            return self.__get_mask(3, 1, 1, False).astype(int)
        elif self.collection == 2:

            return self.__get_mask(8, 3, conf, cumulative).astype(int)
    
    def __get_mask(self, bit_loc, bit_len, value, cumulative):
        """Get mask with specific parameters.

        Args:
            bit_loc (int): Location of the specific QA bits in the value string
            bit_len (int): Length of the specific QA bits
            value (int): A value indicating the desired condition
            cumulative (bool): A boolean value indicating whether to get masker with confidence value larger than the given one.

        Returns:
            np.array: A two-dimension binary mask.
        """

        pos_value = bit_len << bit_loc
        con_value = value << bit_loc

        if cumulative:
            mask = (self.band_data & pos_value) >= con_value
        else:
            mask = (self.band_data & pos_value) == con_value

        return mask    