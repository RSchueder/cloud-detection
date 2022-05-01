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
conda activate cloud-detection
```
**Hardware Environment**
* CUDA 11.6
* CUDNN 8.3
* RTX 2070 Super, 8 GB VRAM

# Running
Use the demo notebook `detect_clouds_tensorflow.ipynb`. You must have unzipped the SPARCS data into the `./data` folder


# Improvements/Next Steps
* Need k-fold validation (would use k=5 and ensure each image ends up in a single fold test set)
* Would employ hyperparameter optimization
* Different achitectures could be examined (for example, Mask R-CNN)
* Ensemble could be created using multiple networks
* Confirm that CNN performs better as compared to pixel-based classifiers
* Could try different loss functions like stochatic gradient descent
* Some labels look quite poor, figure out when these are the culprit
* Identify under which conditions the model performs most poorly
* Test the elimination of classes that are not clouds or cloud shadows
* Data augmentation (more examples)!
* Learning rate decay tuning
* Fix the missing edge of all scenes
* Make implementation more OOP
* Make tests
