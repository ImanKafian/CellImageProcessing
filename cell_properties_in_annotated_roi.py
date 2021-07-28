'''
About: Python script to estimate the cumulative depth-wise geometrical properties of the annotated cells from the binary annotated images.
Author: Iman Kafian-Attari
Date: 28.07.2021
Licence: MIT
version: 0.1
=========================================================
How to use:
1. Select the input image folder containing the annotated images.
2. Select the output folder to store the data.
3. Input the pixel to um ratio.
=========================================================
Notes:
1. This script is meant to perform a cumulative depth-wise cell analysis for binary annotated cell images.
2. The following python packages are required to run this program (dependencies):
   - numpy
   - pandas
   - atplotlib
   - skimage
   - scipy
   - tifffile
   - tk
2. It requires the following inputs from the user:
   - input folder contatining the binary annotated images.
   - output folder to store the data.
   - pixel to um ratio.
4. It automatically reads the images and estimates the following properties of cells with um unit or unitless on an individual and collective basis:
   a) Individual properties:
      - count
      - area (min, max, average)
      - perimeter (min, max, average)
      - eccentricity (min, max, average)
      - orientation (min, max, average)
   b) Collective properties
      - count
      - area
      - perimeter
      - eccentricity
      - orientation
      - minor axis length
      - major axis length
      - distance within cells (average, standard deviation)
5. The output files are saved as pandas dataframe containing cummulative depth-wise geometrical information of the annotated cells.
=========================================================
TODO for version O.2
1. Modify the code in a functional form.
2. Modify the code to work with colored images with different image format.
3. Modify the code to perform blob detection for columnar ROIs as well.
=========================================================
'''

print(__doc__)

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tkinter
from tkinter import filedialog

from skimage import measure
from skimage import io
from skimage.segmentation import clear_border

from scipy.spatial.distance import euclidean

# Building up the main directories
root = tkinter.Tk()
root.withdraw()

# Selecting the Input and Output folders
input_dir = filedialog.askdirectory(parent=root, title='Select the directory containing the annotated images')
output_dir = filedialog.askdirectory(parent=root, title='Select the output directory')
images = sorted(os.listdir(input_dir), key=lambda x: int("".join([i for i in x if i.isdigit()])))

# Making a progress bar for image processing
img_count = len(images)
count = 0

# Inputting Pixel to micrometer ratio
tmp_scale = input('Input the pixel to um ratio...\nThe default value is 0.294'
                  '\nIf the defualt value is okay, press Enter.\nInput --> ')

# Instantiating the scale
if tmp_scale == '':
    scale = 0.294
else:
    scale = int(tmp_scale)

for label in images:

    # Reading the annotated image and its dimensions
    image = io.imread(os.path.join(input_dir, label)) > 0
    y = image.shape[0]

    # Extracting and rebuilding the name of the annotated ROI
    name = label.split('.')[0]

    # Making a progress bar
    print('\n+++++++++++++++++++++++++++++++++++++++++++++++++')
    print(f'\nImage {count + 1} out of {img_count}\n')
    print(name)

    # Making sure that there is an output folder for the image to store the result
    if not os.path.exists(f'{output_dir}\\{name}-CellProperties'):
        os.mkdir(f'{output_dir}\\{name}-CellProperties')

    # Showing the annotated ROI
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.show(block=False)
    plt.pause(3)
    plt.close()

    # Creating a dictionary to record the properties of the cells extracted from the image
    # Individual properties
    individual_properties = {'label': ['10% Thickness', '20% Thickness', '30% Thickness', '40% Thickness',
                                       '50% Thickness', '60% Thickness', '70% Thickness', '80% Thickness',
                                       '90% Thickness', '100% Thickness'],
                           'count': [], 'area_min': [], 'area_max': [], 'area_avg': [],
                           'perimeter_min': [], 'perimeter_max': [], 'perimeter_avg': [],
                           'eccentricity_min': [], 'eccentricity_max': [], 'eccentricity_avg': [],
                           'orientation_min': [], 'orientation_max': [], 'orientation_avg': []}

    # Collective properties
    collective_properties = {'label': ['10% Thickness', '20% Thickness', '30% Thickness', '40% Thickness',
                                       '50% Thickness', '60% Thickness', '70% Thickness', '80% Thickness',
                                       '90% Thickness', '100% Thickness'],
                             'count': [], 'area': [], 'perimeter': [], 'eccentricity': [], 'orientation': [],
                             'minor_axis': [], 'major_axis': [], 'distance_avg': [], 'distance_std': []}

    # Reading the image based on 10% of its thickness and extracting cell properties
    for i in range(10):

        # Reading the image based on its thickness via an accumulative approach
        tile = image[:math.floor((i+1)*0.1*y), :]
        y_tile, x_tile = tile.shape

        # Removing the cells touching the image border!
        tile = clear_border(tile)

        # Showing the tile without border-touching cells
        plt.figure()
        plt.imshow(tile, cmap='gray')
        plt.show(block=False)
        plt.pause(3)
        plt.close()

        # Specifying the white pixels as cell labels
        labeled_tile = measure.label(tile, connectivity=tile.ndim)

        # Extracting the properties of the cell from the labeled tile
        properties = pd.DataFrame(measure.regionprops_table(labeled_tile, tile, properties=['label', 'area', 'perimeter',
                  'eccentricity', 'orientation', 'minor_axis_length', 'major_axis_length', 'centroid'], separator='\t'))

        # Estimating the distnce between cells (average and variation)
        coord_y = np.round(properties['centroid\t0'].values, 0).astype(int)
        coord_x = np.round(properties['centroid\t1'].values, 0).astype(int)

        # Estimating the euclidean distance between cells
        distance = []
        for j in range(coord_y.shape[0]):
            for k in range(coord_y.shape[0]):
                if k >= j:
                    distance.append(float(euclidean([coord_y[j], coord_x[j]], [coord_y[k], coord_x[k]])))
        # Removing the distance=0 to make sure adjacent cells are not included
        distance.remove(float(0))
        # If the distance list is empty, it means that there was only one cell. Thus, add 0 to the distance list
        if not distance:
            distance.append(float(0))

        # Converting the cell properties from pixel unit to um unit!
        properties['normalized_area'] = properties['area']/(x_tile*y_tile)
        properties['normalized_perimeter'] = properties['perimeter']/(x_tile*y_tile*scale)
        properties['area'] = properties['area']*(scale**2)
        properties['perimeter'] = properties['perimeter']*scale
        properties['minor_axis_length'] = properties['minor_axis_length']*scale
        properties['major_axis_length'] = properties['major_axis_length']*scale

        # Saving the cell properties for future analysis
        properties.to_csv(f'{output_dir}\\{name}-CellProperties\\{name}-{(i + 1) * 10}Thick-CellProperties.csv', sep='\t')

        # Storing the depth-dependent individual cell properties
        # Count
        individual_properties['count'].append(np.nanmax(properties['label'].values).ravel())
        # Area
        individual_properties['area_min'].append(np.nanmin(properties['normalized_area'].values).ravel()[0])
        individual_properties['area_max'].append(np.nanmax(properties['normalized_area'].values).ravel()[0])
        individual_properties['area_avg'].append(np.nanmean(properties['normalized_area'].values).ravel()[0])
        # Perimeter
        individual_properties['perimeter_min'].append(np.nanmin(properties['normalized_perimeter'].values).ravel()[0])
        individual_properties['perimeter_max'].append(np.nanmax(properties['normalized_perimeter'].values).ravel()[0])
        individual_properties['perimeter_avg'].append(np.nanmean(properties['normalized_perimeter'].values).ravel()[0])
        # Eccentricity
        individual_properties['eccentricity_min'].append(np.nanmin(properties['eccentricity'].values).ravel()[0])
        individual_properties['eccentricity_max'].append(np.nanmax(properties['eccentricity'].values).ravel()[0])
        individual_properties['eccentricity_avg'].append(np.nanmean(properties['eccentricity'].values).ravel()[0])
        # Orientation
        individual_properties['orientation_min'].append(np.nanmin(properties['orientation'].values).ravel()[0])
        individual_properties['orientation_max'].append(np.nanmax(properties['orientation'].values).ravel()[0])
        individual_properties['orientation_avg'].append(np.nanmean(properties['orientation'].values).ravel()[0])

        # Storing the depth-dependent collective cell properties
        collective_properties['count'].append(np.nanmax(properties['label'].values).ravel()[0]) # Count
        collective_properties['area'].append(np.nansum(properties['normalized_area'].values).ravel()[0]) # Area
        collective_properties['perimeter'].append(np.nansum(properties['normalized_perimeter'].values).ravel()[0]) # Perimeter
        collective_properties['eccentricity'].append(np.nanmean(properties['eccentricity'].values).ravel()[0]) # Eccentricity
        collective_properties['orientation'].append(np.nanmean(properties['orientation'].values).ravel()[0]) # Orientation
        collective_properties['minor_axis'].append(np.nanmean(properties['minor_axis_length'].values).ravel()[0]) # Minor axis length
        collective_properties['major_axis'].append(np.nanmean(properties['major_axis_length'].values).ravel()[0]) # Major axis length
        collective_properties['distance_avg'].append(np.nanmean(distance)*scale) # Distance average
        collective_properties['distance_std'].append(np.nanstd(distance)*scale) # Distance std

        # Reporting the progress
        print(f'{name}: {(i+1)*10}% of thickness is done.')

    # Saving the overall depth-dependent individual and collective cell properties
    individual_properties_df = pd.DataFrame.from_dict(individual_properties, orient='columns')
    individual_properties_df.to_csv(f'{output_dir}\\{name}-CellProperties\\{name}-Overall-Normalized-Individual-CellProperties.csv', sep='\t', header=True, index_label='label')

    collective_properties_df = pd.DataFrame.from_dict(collective_properties, orient='columns')
    collective_properties_df.to_csv(f'{output_dir}\\{name}-CellProperties\\{name}-Overall-Normalized-Collective-CellProperties.csv', sep='\t', header=True, index_label='label')

    count += 1
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')
