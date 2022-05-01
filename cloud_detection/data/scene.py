import os
from pathlib import Path

import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler

from cloud_detection.config import DATA_ROOT, standard_image

standard_data = xr.open_dataset(str(standard_image.absolute()))
cube = standard_data.band_data.values
scalers = dict()
for val_slice in range(0, cube.shape[0]):
    scaler = StandardScaler().fit(cube[val_slice, :, :].flatten().reshape(-1, 1))
    scalers[val_slice] = scaler


class Scene:
    """
    Class to implement Landsat 8 scene data structure.
    """

    def __init__(self, path):
        """
        Args:
            path (str): Path to landsat 8 scene file (.tif)
        """
        self.raw_data = xr.open_dataset(path)
        self.metadata_file = Path(os.path.join(DATA_ROOT, path.stem.split('_')[0] + '_mtl.txt'))
        self.toa_dict = self._load_mtl()
        self.data = self._apply_correction()
        self.geodata = xr.Dataset(data_vars=dict(band_data=(["band", "y", "x"], self.data)),
                                  coords=self.raw_data.coords, attrs=self.raw_data.attrs)
        self.geodata.rio.write_crs(self.raw_data.rio.crs)

    def _load_mtl(self):
        """
        Parses the scene's .mtl metadata file.

        Returns:
            Top of atmosphere reflectance correction parameter dictionary.
        """
        toa_dict = dict()

        with open(self.metadata_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if 'REFLECTANCE_MULT_BAND' in line:
                line = line.rstrip()
                band = int(line.split('_')[3].split(' ')[0])
                mult_value = float(line.split('=')[1])
                toa_dict[band] = [mult_value]
            if 'REFLECTANCE_ADD_BAND' in line:
                line = line.rstrip()
                band = int(line.split('_')[3].split(' ')[0])
                add_value = float(line.split('=')[1])
                toa_dict[band].append(add_value)
            if 'SUN_ELEVATION' in line:
                line = line.rstrip()
                se = float(line.split('=')[1].rstrip())
                sz = 90 - se
                toa_dict['se'] = se
                toa_dict['sz'] = sz

        return toa_dict

    def _apply_correction(self):
        """
        Applies top of atmosphere reflectance correction using
        https://www.usgs.gov/landsat-missions/using-usgs-landsat-level-1-data-product

        Returns:
            TOA reflectance array
        """
        cube = self.raw_data.band_data.values
        corrected_cube = np.zeros(cube.shape)
        for val_slice in range(1, cube.shape[0]):
            vals = cube[val_slice - 1, :, :].copy()
            correction1 = (self.toa_dict[val_slice][0] * vals) + self.toa_dict[val_slice][1]
            correction2 = correction1 / np.sin(np.deg2rad(self.toa_dict['se']))
            # correction3 = scalers[val_slice-1].transform(correction2.flatten().reshape(-1,1))
            # correction3 = correction3.reshape(vals.shape)
            correction3 = correction2
            corrected_cube[val_slice - 1, :, :] = correction3

        return corrected_cube
