import xarray as xr
import xrspatial.multispectral as ms


def get_l8_color(file):
    data = xr.open_dataset(file)
    color = ms.true_color(data.sel({'band': 4}).band_data, 
                          data.sel({'band': 3}).band_data, 
                          data.sel({'band': 2}).band_data)   
    return color

def get_mask_array(file):
    mask = xr.open_dataset(file)
    mask_plot = mask.sel({'band': 1}).band_data.values

    return mask_plot