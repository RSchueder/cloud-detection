import os
from pathlib import Path

from matplotlib import colors


# data parameters
DATA_ROOT = 'data'
TARGET_ROOT = os.path.join(DATA_ROOT, 'target')
SAMPLE_ROOT = os.path.join(DATA_ROOT, 'samples')

data_root = Path(DATA_ROOT)

target_root = Path(TARGET_ROOT)
if not target_root.exists():
    target_root.mkdir(parents=True)

sample_root = Path(SAMPLE_ROOT)
if not sample_root.exists():
    sample_root.mkdir(parents=True)

# modelling parameters
IMG_SIZE = (128, 128)
NUM_BANDS = 10
NUM_CLASSES = 7  # https://www.usgs.gov/landsat-missions/spatial-procedures-automated-removal-cloud-and-shadow-sparcs-validation-data
INPUT_SHAPE = (list(IMG_SIZE) + [NUM_BANDS])
BATCH_SIZE = 16
EPOCHS = 50

# plotting parameters
custom_cmap = colors.ListedColormap(['black', 'grey', 'blue', 'pink', 'green', 'white', 'red'])
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
custom_norm = colors.BoundaryNorm(bounds, custom_cmap.N)

custom_cmap_simple = colors.ListedColormap(['black', 'grey', 'white', 'green'])
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
custom_norm_simple = colors.BoundaryNorm(bounds, custom_cmap_simple.N)

# misc
standard_image = Path(r'data\LC80010812013365LGN00_18_data.tif')
