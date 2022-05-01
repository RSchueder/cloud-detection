# cloud-detection
This is a repository demonstrating how to detect clouds in Landsat 8 images via semantic segmentation and a UNET CNN. 
We use the [SPARCS Dataset](https://www.usgs.gov/landsat-missions/spatial-procedures-automated-removal-cloud-and-shadow-sparcs-validation-data) to train and validate the model.

# Getting started
Ideally this would be run in a docker container, but I do not have Windows Professional on my personal computer, and so 
I cannot activate Hyper-V. As for using docker through WSL, CUDA is embarassingly slow via WSL, and so is not appropriate 
for GPU-based neural net computations. So we use conda in Windows.

You can obtain the project environment via:
```
conda env create -f requirements.yml
conda activate mpc-mcd
```

# Running
Use the demo notebook `detect_clouds_tensorflow.ipynb`.

# Improvements
* need k-fold validation
* different achitectures could be examined (Mask R-CNN)
* SGD loss function?
* Some labels look quite por
* should I eliminate some of the other classes?
* Data augmentation (more examples)!
* Learning rate decay
* would be good to fix the missing edge of all frames

# Hardware Environment
* CUDA 11.6
* CUDNN 64.8
* RTX 2070 Super, 8 GB VRAM

