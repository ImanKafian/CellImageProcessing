'''
About: Python script to detect cells from the extracted ROIs via blob detection technique.
Author: Iman Kafian-Attari
Date: 28.07.2021
Licence: MIT
version: 0.1
=========================================================
How to use:
1. Select the input image folder containing input images and ROI subfolders.
=========================================================
Notes:
1. This script is meant to perform a basic blob detection scheme optimized for cell detection.
2. The following python packages are required to run this program (dependencies):
   - Numpy
   - Pandas
   - OpenCV (= cv2)
   - tifffile
   - tk
2. It requires the following inputs from the user:
   - input folder contatining images and ROI subfolders
3. The input images must be in '.tif' file format. Otherwise change the following lines to your desired format (.*):
   - Line 53
   - Line 116 & 128
4. It automatically reads the images and its square ROIs.
5. It performs blob detection for each square ROI and stores the number of detected cells in a numpy array per image.
6. The output files are saved as pandas dataframe containing information on the number of cells per ROI per image.
=========================================================
TODO for version O.2
1. Modify the code in a functional form.
2. Modify the code to work with colored images with different image format.
3. Modify the code to perform blob detection for columnar ROIs as well.
=========================================================
'''

print(__doc__)

import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import tifffile
import os

import tkinter
from tkinter import filedialog

# Building up the main directories
root = tkinter.Tk()
root.withdraw()

# Selecting the folder with all image subfolders and listing them for iteration
input_dir = filedialog.askdirectory(parent=root, title='Choose the input directory (The folder with all input images and ROI subfolders)')
img_files = [file for file in sorted(os.listdir(input_dir), key=lambda x: int("".join([i for i in x if i.isdigit()]))) if '.tif' in file]

# Making a progress bar for image processing
img_count = len(img_files)
count = 0

# Instantiating the blob detector:
blob_params = cv2.SimpleBlobDetector_Params()

# Assigning the parameters to the blob object
# Specifying the thresholds
blob_params.minThreshold = 0
blob_params.maxThreshold = 225
# Filtering by color
blob_params.filterByColor = True
blob_params.blobColor = 0.3 # 1: White, 0: Black
# Filtering by circularity of the cells (features)
blob_params.filterByCircularity = True
blob_params.minCircularity = 0.1
blob_params.maxCircularity = 1
# filtering by area
blob_params.filterByArea = True
blob_params.minArea = 30
blob_params.maxArea = 4000
# filtering by convexity
blob_params.filterByConvexity = True
blob_params.minConvexity = 0.1
blob_params.maxConvexity = 1
# filtering by Inertia
blob_params.filterByInertia = True
blob_params.minInertiaRatio = 0.1
blob_params.maxInertiaRatio = 1
# Minimum distance between features neighbor to each other
blob_params.minDistBetweenBlobs = 15

# Setting up the block detector  with the parameters
cell_detector = cv2.SimpleBlobDetector_create(blob_params)

# Accessing the images
for image in img_files:

    print('\n+++++++++++++++++++++++++++++++++++++++++++++++++')
    print(f'\nImage {count + 1} out of {img_count}\n')
    print(image[:-4])

    # Making a directory for square ROI blob-detected tiles
    if not os.path.exists(f'{os.path.join(input_dir, image[:-4])}\\{image[:-4]}-Square-ROIs-BlobDetection'):
        os.mkdir(f'{os.path.join(input_dir, image[:-4])}\\{image[:-4]}-Square-ROIs-BlobDetection')

    # Accessing the square ROI tiles
    img_path = os.path.join(input_dir, image[:-4], f'{image[:-4]}-Square-ROIs')

    # Getting the number of tiles row-wise and column-wise per image from the last image
    img = image.split('-')
    row = int(img[-1].split('.')[0])
    col = int(img[2])

    # Creating a numpy array to store the number of blob detected in each ROI tile for each image
    blob_count = np.zeros((row+1, col+1))

    # Reading ROI tiles
    for j in range(col):
        for i in range(row):
            roi = tifffile.imread(f'{img_path}\\{image[:-4]}-Col-{j + 1}-Row-{i + 1}.tif', key=0)

            # Detecting the cells in the image
            key_pts = cell_detector.detect(roi)
            print(key_pts)
            print(f'Number of detected cells in the {image[:-4]} (Col={j + 1}, Row={i + 1}): {len(key_pts)}')
            blob_count[i, j] = len(key_pts)

            # Saving the ROI tiles with the detected blobs
            roi_blob = cv2.drawKeypoints(roi, key_pts, np.array([]), (255, 0, 0),
                                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            plt.imshow(roi_blob)
            plt.imsave(f'{os.path.join(input_dir, image)}\\{image[:-4]}-Square-ROIs-BlobDetection\\Blob-{image[:-4]}-Col-{j + 1}-Row-{i + 1}.tif', roi_blob, format='tiff', dpi=300)

    # Averaging the number of detected objects in each row of the image
    for i in range(row):
        blob_count[i, -1] = np.nanmean(blob_count[i, :-1])

    # Summation of detected blobs over all image rows including the average column
    for j in range(col+1):
        blob_count[-1, j] = np.nansum(blob_count[:-1, j])

    # Saving the blob_count array as a pandas dataframe
    index = [i+1 for i in range(row)]
    index.append('Total')
    cols = [i+1 for i in range(col)]
    cols.append('Average')
    data = pd.DataFrame(data=blob_count.astype(int), index=index, columns=cols, dtype=int)
    data.to_csv(f'{os.path.join(input_dir, image[:-4])}\\{image[:-4]}-Square-ROIs-BlobDetection\\Blob-{image[:-4]}.csv', sep='\t')

    count += 1
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')
