'''
About: Python script to make square and columnar ROIs from grayscale/single-channel high-resolution cell images.
Author: Iman Kafian-Attari
Date: 28.07.2021
Licence: MIT
version: 0.1
=========================================================
How to use:
1. Select the input image folder.
2. Specify the width of your ROI in pixel.
3. Input the sample's label.
4. Input the sample's thickness in mm.
5. Input the list of strains used in the experiment, separate the values with comma.
6. Repeat the process for the remaining samples, if needed.
=========================================================
Notes:
1. This script is meant to prepare ROIs for cell segmentation.
2. The following python packages are required to run this program (dependencies):
   - Numpy
   - Matplotlib
   - tifffile
   - tk
2. It requires the following inputs from the user:
   - input folder,
   - ROI width in pixel
3. The input images must be in '.tif' file format. Otherwise change the following lines to your desired format (.*):
   - Line 59
   - Line 134 & 139
4. It automatically reads the images and extract the square and columnar ROIs from the images.
   The columnar ROI images have the user-defined width and height of the image,
   while the square ROIs have the user-defined length.
5. It stores the following data per image:
   - Square ROIs covering the image
   - Columnar ROIs covering the image
6. The number of ROIs are estimated automatically.
7. The output ROIs are saved as grayscale tiff images.
=========================================================
TODO for version O.2
1. Modify the code in a functional form.
2. Modify the code to work with colored images with different image format.
=========================================================
'''

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import os
import tifffile
import tkinter
from tkinter import filedialog

# Building up the main directories
root = tkinter.Tk()
root.withdraw()

# Selecting the input image folder and listing the images for iteration
input_dir = filedialog.askdirectory(parent=root, title='Choose the input image directory (The folder containin all the images)')
img_files = [file for file in sorted(os.listdir(input_dir), key=lambda x: int("".join([i for i in x if i.isdigit()]))) if '.tif' in file]

# Making a progress bar for image processing
img_count = len(img_files)
count = 0

# Inputting the width of the ROI
tmp_roi_width = input('Input the desired width for the ROI in pixel...\nThe default value is 1024 pixels.'
                      'If the default value is okay just press Enter.\nInput --> ')

# Instantiating the ROI width
if tmp_roi_width == '':
    roi_width = 1024
else:
    roi_width = int(tmp_roi_width)

# Accessing the images
for image in img_files:

    # Making a directory for the square ROI tiles
    if not os.path.exists(f'{input_dir}\\{image[:-4]}-Square-ROIs'):
            os.mkdir(f'{input_dir}\\{image[:-4]}-Square-ROIs')

    # Making a directory for the columnar ROIs-Suitable for annotation
    if not os.path.exists(f'{input_dir}\\{image[:-4]}-Columnar-ROIs'):
            os.mkdir(f'{input_dir}\\{image[:-4]}-Columnar-ROIs')

    # Reading the image as BigTiff file --> Images should be grayscale or the first channel will be written only!
    img_path = os.path.join(input_dir, image)
    img = tifffile.imread(f'{img_path}', key=0)[:, :, 0]

    print('\n+++++++++++++++++++++++++++++++++++++++++++++++++')
    print(image[:-4])

    # Showing the original image
    plt.figure()
    plt.imshow(img)
    plt.show(block=False)
    plt.pause(3)
    plt.close()

    print(f'\nImage {count + 1} out of {img_count}\n')

    # Accessing image dimensions to make the ROI tesselation --> Each pixel is equal to 1024 px
    # Accessing the relevant directories for storing the ROIs
    roi_tiles_path = os.path.join(input_dir, f'{image[:-4]}-Square-ROIs')
    roi_columnar_path = os.path.join(input_dir, f'{image[:-4]}-Columnar-ROIs')

    # Square ROI dimension (X, Y): (roi_width px, roi_width px), default values = (1024 px, 1024 px)
    print(f'{image[:-4]} dimensions (Y, X): ', img.shape)
    y = img.shape[0]
    x = img.shape[1]

    # Estimating the number of ROI tiles based on the image dimensions
    count_roi_y = np.floor_divide(y, roi_width).astype(int)
    count_roi_x = np.floor_divide(x, roi_width).astype(int)
    print(f'{image[:-4]} tiles in y: ', count_roi_y)
    print(f'{image[:-4]} tiles in x: ', count_roi_x)
    print(f'{image[:-4]}-Total ROI tiles: ', count_roi_x*count_roi_y)
    # Estimating the remainder for extracting the ROIs from the central part of the image
    remainder_roi_x = np.floor(np.remainder(x, roi_width)).astype(int)
    img = img[:, np.floor(remainder_roi_x/2).astype(int):]

    # Showing the original image
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    # Extracting ROIs from the image
    for i in range(count_roi_x):

        # Extracting columnar ROIs for annotation
        roi_columnar = img[:, i*roi_width:(i+1)*roi_width]
        tifffile.imwrite(f'{roi_columnar_path}\\{image[:-4]}-Col-{i + 1}.tif', roi_columnar)

        # Extracting the ROI tiles from the ith columnar ROI
        for j in range(count_roi_y):
            roi = roi_columnar[j*roi_width:(j+1)*roi_width, :]
            tifffile.imwrite(f'{roi_tiles_path}\\{image[:-4]}-Col-{i + 1}-Row-{j + 1}.tif', roi)

    count += 1
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')
