import rasterio as rio
import pandas as pd
from osgeo import osr
from math import floor
from rasterio.windows import Window
import numpy as np
import cv2
import random


def image_coords(df, trans):
    """
    A function to determine the (x, y) image coordinates for each labelled bird.
    :param df: A Pandas DataFrame containing labelled points for birds in a GeoTiff
    :param trans: The reverse the transformation to convert latitude and longitude to pixel (x, y) coordinates
    :return: The original Pandas DataFrame with two extra columns, "x" and "y" which contain the x & y image
    coordinates respectively.
    """
    for i, r in df.iterrows():
        # Determine the ortho coordinates
        lat, lon, _ = transform.TransformPoint(r.POINT_X, r.POINT_Y)

        # Determine the pixel coordinates
        x, y = trans * (lon, lat)

        # Write the x and y coordinates to the DataFrame
        df.loc[i, "x"] = floor(x)
        df.loc[i, "y"] = floor(y)

    # Return the DataFrame which now contains the (x, y) image coordinates
    return df


def bird_labels(min_x, min_y, df, row, folder, box_sz, classes, negative=False):
    """
    A function to extract bounding boxes and labels for birds in images in format for use with YOLO.
    :param min_x: The most left pixel x coordinates of an image
    :param min_y: The top pixel y coordinate of an image
    :param df: A Pandas DataFrame containing labelled points for birds in a GeoTiff
    :return: A list of strings in the format "<object-class> <x> <y> <width> <height>" as specified here
    https://pjreddie.com/darknet/yolo/
    """

    # Create an empty list to append label strings to
    labels = []

    path = folder + row.speciesName.replace(" ", "") + "_" + str(index) + ".jpg"

    label = path
    positive = []
    # Iterate through the DataFrame to determine the labels which fall within the image
    for r in df.itertuples():
        if min_x < r.x - 30 and r.x + 30 < min_x + 416 and min_y < r.y - 30 and r.y + 30 < min_y + 416:
            # determine centre pixel coordinates for the labelled bird wrt the total image height and width
            xmin = max(int(r.x - (0.5 * box_sz) - min_x), 0)
            ymin = max(int(r.y - (0.5 * box_sz) - min_y), 0)
            xmax = min(int(r.x + (0.5 * box_sz) - min_x), img_sz)
            ymax = min(int(r.y + (0.5 * box_sz) - min_y), img_sz)

            cls = classes.index(r.speciesName)

            positive.append((r.x - min_x, r.y - min_y))

            # Write the label details to a string
            label = label + " " + str(xmin) + "," + str(ymin) + "," + str(xmax) + "," + str(ymax) + "," + str(cls)

    if negative:
        print(len(positive))


    # Append the string to the list defined at beginning of function.
    label = label + "\n"
    labels.append(label)

    # Return list containing labels for birds in the image
    return labels


# Define output image size
img_sz = 416
# Define box size
box_sz = 50

# Define output folder
folder = "D:/MASTERS/Data/Birds/DAT/lf_cn_mb_rfb/yolov3/" + str(box_sz) + "x" + str(box_sz) + "/all/"

# List birds of interest
birds_of_interest = ["Lesser Frigatebird", "Common Noddy", "Masked Booby", "Red-Footed Booby"]
# birds_of_interest = ["Masked Booby", "Red-Footed Booby"]

# Define the file containing bird labels and locations, then read the file into a pandas dataframe
bird_points = 'D:/MASTERS/Data/Birds/RAW/2018-06-28/Labels/RAW_RaineIsland-BirdCount-MGA55_20180628.csv'
birds = pd.read_csv(bird_points)

# Define the ESPG code for the label coordinates
source = osr.SpatialReference()
source.ImportFromEPSG(28355)

# Open the orthoimage
print("Opening Orthoimage...")
with rio.open('D:/MASTERS/Data/Birds/RAW/2018-06-28/GeoTIFF/RAW_RaineIsland-Ortho-1cm-rectified-COG_20180628.tif') as raine_island:
    # Identify the affine transformation between pixel (x, y) coordinates and latitude and longitude
    fwd = raine_island.transform
    # Reverse the transformation so latitude and longitude can be converted to pixel (x, y) coordinates
    rev = ~fwd

    # Determine the EPSG code for the orthoimage
    crs = int(raine_island.crs.to_string()[5:])
    target = osr.SpatialReference()
    target.ImportFromEPSG(crs)

    # Create the transform object to transform from the label coordinates to the ortho coordinates
    transform = osr.CoordinateTransformation(source, target)

    # For each bird:
    print("Finding birbs...")
    birds = image_coords(birds, rev)

    birds = birds[birds.speciesName.isin(birds_of_interest)]

    bird_types = list(birds.speciesName.unique())
    print(bird_types)
    image_labels = []

    for index, row in birds.iterrows():
        # Define the top left pixel coordinates for a 416 * 416 pixel image centered on the bird
        x_min = row.x - img_sz/2
        y_min = row.y - img_sz/2

        # Define the red, green and blue layers for the 416 * 416 image centered on the bird
        img_r = raine_island.read(1, window=Window(x_min, y_min, img_sz, img_sz))
        img_g = raine_island.read(2, window=Window(x_min, y_min, img_sz, img_sz))
        img_b = raine_island.read(3, window=Window(x_min, y_min, img_sz, img_sz))

        # Stack the red, green and blue images to create one RGB image
        img = np.dstack((img_r, img_g, img_b))

        # Call bird_labels to create the labels for each image
        image_labels.extend(bird_labels(x_min, y_min, birds, row, folder, box_sz, birds_of_interest, True))

        # # Write the image to a .jpg file with the naming convention of birdSpecies_index.jpg
        cv2.imwrite(folder + row.speciesName.replace(" ", "") + "_" + str(index) + ".jpg", img)

    open(folder + "_annotations.txt", "w").writelines(image_labels)

    # Write the names of the classes to a .txt file
    with open(folder + "_classes.txt", "w") as f:
        for c in bird_types:
            f.write(c.replace(" ", "") + "\n")

# Close the GeoTiff file
raine_island.close()

# Print statement to signify that the script has finished running
print("Birb extraction complete!")
