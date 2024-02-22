# Takes in large geotiff and a csv
# Crops in geotiff in a sliding window manner, saving each image with associated labels from csv in different folders
# Written by: Serena Mou, based on some code from Jaime | 3 Jan, 2024
# Modified by: Matthew Tavatgis | 26 Jan, 2024
# Install osgeo through pip3 install gdal | may require gdal==3.0.4 on 20.04, as well as downgrading setuptools==57.5.0 (can reupgrade after gdal install)

# TODO
# - Normalise image size to fixed altitude based on coordinates

import os
import sys
import shutil
import argparse
from distutils.util import strtobool

import cv2
import numpy as np
import pandas as pd
import rasterio as rio
from osgeo import osr
from osgeo import ogr
from math import floor
from rasterio.windows import Window

def arg_parse():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='slice and dice baby')

    parser.add_argument("--src", dest = "src_dir",
            help = "Source files to operate on", default = None, type = str)
    parser.add_argument("--csv", dest = "src_csv",
            help = "Provide a custom source csv", default = None, type = str)
    parser.add_argument("--name", dest = "output_name",
            help = "Output label", default = None, type = str)
    parser.add_argument("--birds", dest = "use_birds",
            help = "Select specific labels to include", nargs='*', default = None, type = str)
    parser.add_argument("--show", dest = "show",
            help = "Optionally show window preview of outputs", default = False, type = lambda x: bool(strtobool(x)))
    
    return parser.parse_args()

class CutGeotiff():

    def __init__(self, args):
        
        # Weclome
        print("\nSlice and Dice\n")

        # Define output image size
        self.img_sz = 512
        overlap_percent = 10
        self.overlap = round(self.img_sz*overlap_percent/100)
        self.iter_size = round(self.img_sz-self.overlap)

        # Define box size
        self.box_sz = 50
        self.border = int(0.2*self.box_sz)
        print(self.border)
        print(self.overlap)
        print(self.iter_size)

        # Threshold for empty image
        self.black_thresh = 0.5
        
        # Colours
        self.colours = [(0,110,170),(128,128,128),(75,25,230),(128,0,0),(25,230,230),(180,30,145),(60,180,60),(212,190,250)]
        
        # Config
        self.SHOW = args.show
        self.WRITE = True
        self.DEBUG = False
        
        # Check input name
        self.output_name = args.output_name
        if self.output_name is None:
            print("ERROR: Output name must be specified\nEXITING...")
            exit()

        # Define output folder
        folder_name = "{}_sz{}_overlap{}_box{}_border{}".format(self.output_name, self.img_sz, overlap_percent, self.box_sz, self.border)
        sub_dirs = ['images', 'labels', 'images_labelled']
        self.folder = os.path.join(args.src_dir, folder_name)
        
        # Info
        print("Writing to: {}\n".format(folder_name))

        # Create output directories
        if self.WRITE:
            
            # Check if path exists
            if os.path.isdir(self.folder):
                
                # Warn if output will be overwritten
                if input("WARNING: Output directory exists, data will be overwritten Y/N?: ").lower() != "y":
                    print("EXITING...\n")
                    exit()
                else:
                    print("CONTINUING...\n")
                    try:
                        shutil.rmtree(self.folder)
                    except OSError as error:
                        print(error)
                        exit()
            
            # Create output paths
            os.mkdir(self.folder)
            for output in sub_dirs:
                full = os.path.join(self.folder, output)
                os.mkdir(full)

        # Find input files
        contents = os.listdir(args.src_dir)

        csv_files = []
        tif_files = []
        for f in contents:
            if f.endswith('.csv'): csv_files.append(f)
            if f.endswith('.tif'): tif_files.append(f)
        
        # Check for custom csv
        if args.src_csv is not None:
            if os.path.exists(args.src_csv):
                csv_files = [args.src_csv]
            else:
                print("Error: Custom CSV file {} not found. Exiting...\n".format(args.src_csv))
                exit()

        if len(csv_files) == 1 and len(tif_files) == 1:
            print("Found:\nCSV: {}\nTIF: {}".format(csv_files[0], tif_files[0]))
        else:
            print("ERROR: Ambiguous data source, source should contain single tif and csv! Found:")
            for f in csv_files: print(f)
            for f in tif_files: print(f)
            print("EXITING...\n")
            exit()

        # Define the file containing bird labels and locations, then read the file into a pandas dataframe
        self.birds = pd.read_csv(os.path.join(args.src_dir, csv_files[0]))
        self.tiff = os.path.join(args.src_dir, tif_files[0])
        
        # Select class labels to care about
        if args.use_birds is not None:
            self.birds_of_interest = args.use_birds
        else:
            self.birds_of_interest = self.get_all_species()

        print("\nExtracting:")
        for bird in self.birds_of_interest: print(bird)

        # Define the ESPG code for the label coordinates
        self.source = osr.SpatialReference()
        #self.source.ImportFromEPSG(28355)
        self.source.ImportFromEPSG(4326)
        
        print("\nSetup Complete!!!\n")

    def get_all_species(self):
        birds = []
        all_birds = self.birds['LAYER'].unique()
        for bird in all_birds:
            birds.append(bird)

        # sort alphabetical
        return sorted(birds)

    def image_coords(self, df, pix_transform, crs_transform, crs_target):
        """
        A function to determine the (x, y) image coordinates for each labelled bird.
        :param df: A Pandas DataFrame containing labelled points for birds in a GeoTiff
        :param pix_transform: The reverse the transformation to convert latitude and longitude to pixel (x, y) coordinates
        :param crs_transform: The ESPG coordinate transform from the label CRS to the target CRS
        :parma crs_target: If the CRS target is 6933 flip lat/lon lol
        :return: The original Pandas DataFrame with two extra columns, "x" and "y" which contain the x & y image
        coordinates respectively.
        """

        for i, r in df.iterrows():
            
            # Determine the ortho coordinates
            # Redundant if lat lon are already in correct format
            lat, lon, _ = crs_transform.TransformPoint(r.LAT, r.LON)
            
            # Determine the pixel coordinates
            if crs_target == 6933:
                x, y = pix_transform * (lat, lon)
            else:
                x, y = pix_transform * (lon, lat)
            
            # Write the x and y coordinates to the DataFrame
            df.loc[i, "x"] = floor(x)
            df.loc[i, "y"] = floor(y)

        # Return the DataFrame which now contains the (x, y) image coordinates
        return df


    def bird_labels(self, min_x, min_y, df, classes):
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
      
        # Iterate through the DataFrame to determine the labels which fall within the image
        for r in df.itertuples():
            if r.x > (min_x+self.border) and r.x < (min_x+self.img_sz-self.border) and r.y > (min_y+self.border) and r.y < (min_y+self.img_sz-self.border):
                
                cx = (r.x-min_x)/self.img_sz
                cy = (r.y-min_y)/self.img_sz
                w = self.box_sz/self.img_sz
                h = self.box_sz/self.img_sz
                
                try:
                    cls = classes.index(r.LAYER)
                    write_str = "%d %0.4f %0.4f %0.4f %0.4f"%(cls,cx,cy,w,h)
                    labels.append(write_str)
                except:
                    pass

                if self.DEBUG:
                    print("debugging")
                    print(r)
                    print(write_str)
                    print(r.x,r.y)
                    input()
          

        # Return list containing labels for birds in the image
        return labels
    
    def run(self):
        
        with rio.open(self.tiff) as full_tiff:

            # Get n columns and rows to use
            columns = floor(full_tiff.width/self.iter_size)
            rows = floor(full_tiff.height/self.iter_size)
            
            # Identify the affine transformation between pixel (x, y) coordinates and latitude and longitude
            fwd = full_tiff.transform
            # Reverse the transformation so latitude and longitude can be converted to pixel (x, y) coordinates
            rev = ~fwd

            # Determine the EPSG code for the orthoimage
            crs = int(full_tiff.crs.to_string()[5:])
            target = osr.SpatialReference()
            target.ImportFromEPSG(crs)

            # Create the transform object to transform from the label coordinates to the ortho coordinates
            transform = osr.CoordinateTransformation(self.source, target)

            # For each bird:
            print("Finding birbs...", end="")
            birds = self.image_coords(self.birds, rev, transform, crs)
            print(" Birds found!")

            # Create a display window
            if self.SHOW:
                cv2.namedWindow("named")

            # Save a file describing classes included
            if self.WRITE:
                with open(os.path.join(self.folder, "classes.txt"), "w") as f:
                    for label in self.birds_of_interest:
                        f.write("{}\n".format(label))
            
            # Iterate over each region
            for u in range(columns):
                print("{}/{} Done     \r".format(u, columns), end="")
                for v in range(rows):

                    # Calculate (top left?) corner of region
                    x_min = int(u*self.iter_size)
                    y_min = int(v*self.iter_size)
                    
                    # Generate label for region
                    label = self.bird_labels(x_min, y_min, birds, self.birds_of_interest)
                    
                    # Get rgb channels fromt tiff
                    img_r = full_tiff.read(1, window=Window(x_min, y_min, self.img_sz, self.img_sz))
                    img_g = full_tiff.read(2, window=Window(x_min, y_min, self.img_sz, self.img_sz))
                    img_b = full_tiff.read(3, window=Window(x_min, y_min, self.img_sz, self.img_sz))
                    img = np.dstack((img_b, img_g, img_r))
                    
                    # number of pixels that are all zeros (out of scope)
                    zero_pix = (sum(img_r+img_g+img_b) == 0).sum()

                    # if the image is all black (out of scope) then skip
                    if zero_pix > int(self.img_sz*self.black_thresh):
                        continue

                    idx_name = 'img-%d-%d'%(u,v)
                    img_wlabel_name = os.path.join(self.folder, 'images_labelled', "{}_{}.png".format(self.output_name, idx_name))
                    img_name = os.path.join(self.folder, 'images', "{}_{}.png".format(self.output_name, idx_name))
                    txt_name = os.path.join(self.folder, 'labels', "{}_{}.txt".format(self.output_name, idx_name))

                    if self.WRITE:
                    
                        if len(label)>0:
                            cv2.imwrite(img_wlabel_name,img)
                            with open(txt_name,'w') as f:
                                for l in label:
                                    f.write("%s\n"%l)
                        else: 
                            cv2.imwrite(img_name, img)
                            with open(txt_name,'w') as f:
                                pass

                    if self.SHOW:
                        if len(label)>0:
                            for l in label:
                                cls,cx,cy,w,h = l.split(' ')
                                cls = int(cls)
                                cx = float(cx)
                                cy = float(cy)
                                top_left = (int(cx*self.img_sz-0.5*self.box_sz),int(cy*self.img_sz-0.5*self.box_sz))
                                bottom_right = (int(cx*self.img_sz+0.5*self.box_sz),int(cy*self.img_sz+0.5*self.box_sz))
                                color = self.colours[cls]
                                cv2.rectangle(img, top_left, bottom_right, color, 5)
                        
                            # Show image and wait for key, exit if 'q'
                            cv2.imshow("named",img)
                            if cv2.waitKey(0) == ord('q'):
                                self.cleanup(full_tiff)
            
            # Clear counter
            sys.stdout.write("\033[K")

        # Cleanup and exit
        self.cleanup(full_tiff)

    def cleanup(self, tiff_file):

        # Close the GeoTiff file
        tiff_file.close()
        cv2.destroyAllWindows()

        # Print statement to signify that the script has finished running
        print("Birb extraction complete!")

        # Exit
        exit()

def main():
    # Read args
    args = arg_parse()
    
    # Run main
    cg = CutGeotiff(args)
    cg.run()

if __name__ == '__main__':
    main()
