# Data Processing

This directory contains modules and notebooks for processing, cleaning, and preparing the datasets used in the drowsiness detection system.

## Directory Contents

### Data Cleaning Notebooks

Four Jupyter notebooks are provided for cleaning and preparing the dataset classes:

- `cleaning_Close_Eye_Class.ipynb`: Processes and cleans images of closed eyes
- `cleaning_Open_Eye_Class.ipynb`: Processes and cleans images of open eyes
- `cleaning_Yawn_Class.ipynb`: Processes and cleans images of yawning mouths
- `cleaning_No_Yawn_Class.ipynb`: Processes and cleans images of non-yawning mouths

These notebooks perform tasks such as:
- Removing low-quality or corrupted images
- Ensuring proper cropping of facial features
- Filtering out irrelevant images
- Ensuring data consistency

### Data Processing Scripts

- `preprocess.py`: Script for cropping bounding boxes around eyes and mouth from full face images

## Dataset Processing

### RoboFlow Dataset to Feature Extraction

The system uses datasets originally collected from RoboFlow, which contain:
- Full face images
- Bounding box annotations for:
  - Left eye
  - Right eye
  - Mouth

The `preprocess.py` script is specifically designed to:
1. Read the full face images with bounding box annotations
2. Extract only the eye and mouth regions using these annotations
3. Save the cropped features as separate images for training the classification models

This extraction process is crucial because:
- Our models only need the specific facial features (eyes and mouth) for classification
- Working with smaller, focused regions improves model efficiency and accuracy
- It reduces unnecessary information that could distract the learning process

## Data Classes

The drowsiness detection system uses four main data classes:

1. **Eye State Classes**:
   - `Close_Eye`: Cropped images of closed eyes
   - `Open_Eye`: Cropped images of open eyes

2. **Mouth State Classes**:
   - `Yawn`: Cropped images of yawning mouths
   - `No_Yawn`: Cropped images of non-yawning mouths

## Data Processing Pipeline

The complete workflow for preparing the dataset:

1. **Collection**: Obtaining labeled datasets from RoboFlow with bounding box annotations
2. **Extraction**: Using `preprocess.py` to crop eyes and mouth regions based on bounding boxes
3. **Cleaning**: Running the notebooks to clean and validate the cropped feature images
4. **Organization**: Structuring the cleaned images into appropriate class directories
5. **Training**: Using the processed images to train the classification models


