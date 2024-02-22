# Takes in large geotiff and a csv
# Crops in geotiff in a sliding window manner, saving each image with associated labels from csv in different folders
# Written by: Serena Mou, based on some code from Jaime | 3 Jan, 2024
# Modified by: Matthew Tavatgis | 26 Jan, 2024 (convert to distance normalised function)
# Install osgeo through pip3 install gdal | may require gdal==3.0.4 on 20.04, as well as downgrading setuptools==57.5.0 (can reupgrade after gdal install)

# TODO
# - optemiiiiiiiiiiiiiise

import os
import sys
import yaml
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
    parser.add_argument("--norm", dest = "norm_img",
            help = "Opitionally normalise image to meters rather than pixels", default = False, type = lambda x: bool(strtobool(x)))
    parser.add_argument("--box_cfg", dest = "box_cfg",
            help = "Optionally apply custom box sizes for each class accoridng to config yaml", default = None, type = str)
    
    return parser.parse_args()

class CutGeotiff():

    def __init__(self, args):
        
        # Weclome
        print("\nSlice and Dice\n")

        # Config
        self.target_px = 512            # Target image output size
        self.overlap_m = 0.2            # Overlap between the cut images in meters
        self.black_thresh = 0.5         # Threshold for empty image
        self.img_sz_m = 4               # Size of image in meters
        self.box_sz_m = [0.4]           # Bounding box size in meters
        self.border_m = 0.1             # Exclusion border around edge of image in meters
        self.norm_img = args.norm_img   # Select whether image is normalised to pixel or meter size

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
        folder_name = "{}_sz{}m_overlap{}m_box{}m_border{}m".format(self.output_name, self.img_sz_m, self.overlap_m, self.box_sz_m[0], self.border_m)
        folder_name = folder_name.replace('.', '')
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

        # Check for custom box sizes
        if args.box_cfg is not None:

            # Load config
            if not os.path.exists(args.box_cfg):
                print("ERROR: Box config '{}' does not exist! EXITING...\n".format(args.box_cfg))
                sys.exit()
            
            box_sizes = []
            with open(args.box_cfg, "r") as stream:
                config = yaml.safe_load(stream)
                
                # Load classes
                for bird in self.birds_of_interest:
                    try:
                        box_sizes.append(config[bird])
                    except:
                        print("ERROR: Label '{}' does not exist in config file '{}'! EXITING...\n".format(bird, args.box_cfg))
                        sys.exit()

            # Update box values if successful
            self.box_sz_m = box_sizes 

        # Info
        print("\nExtracting:")
        if len(self.box_sz_m) == 1:
            for i, bird in enumerate(self.birds_of_interest): print("{} {}".format(i, bird))
        else:
            for i, bird in enumerate(self.birds_of_interest): print("{} {}: {:.2f}m box".format(i, bird, self.box_sz_m[i]))
        
        # Define the ESPG codes for the label coordinates
        InCRS = osr.SpatialReference()
        InCRS.ImportFromEPSG(4326)
        OutCRS = osr.SpatialReference()
        OutCRS.ImportFromEPSG(6933)

        # Create the transform object to transform from the label coordinates to the ortho coordinates
        self.crs_transform = osr.CoordinateTransformation(InCRS, OutCRS)
        
        # Info
        print("\nSetup Complete!!!\n")

    def get_all_species(self):
        birds = []
        all_birds = self.birds['LAYER'].unique()
        for bird in all_birds:
            birds.append(bird)

        # sort alphabetical
        return sorted(birds)

    def image_coords(self, df, pix_transform, crs_transform):
        """
        A function to determine the (x, y) image coordinates for each labelled bird.
        :param df: A Pandas DataFrame containing labelled points for birds in a GeoTiff
        :param pix_transform: The reverse the transformation to convert latitude and longitude to pixel (x, y) coordinates
        :param crs_transform: The ESPG coordinate transform from the label CRS to the target CRS
        :return: The original Pandas DataFrame with two extra columns, "x" and "y" which contain the x & y image
        coordinates respectively.
        """
        # Itearte over dataset rows
        for i, r in df.iterrows():
            
            # Determine the ortho coordinates
            lat, lon, _ = crs_transform.TransformPoint(r.LAT, r.LON)
            
            # Determine the pixel coordinates
            x, y = pix_transform * (lat, lon)
            
            # Write the x and y coordinates to the DataFrame
            df.loc[i, "x"] = floor(x)
            df.loc[i, "y"] = floor(y)

        # Return the DataFrame which now contains the (x, y) image coordinates
        return df

    def bird_labels(self, min_x, min_y, img_size, box_sizes, border, df, classes):
        """
        A function to extract bounding boxes and labels for birds in images in format for use with YOLO.
        :param min_x: The most left pixel x coordinates of an image
        :param min_y: The top pixel y coordinate of an image
        :param img_size: Size of image to label in pixels
        :param box_sizes: Size of boxes to use in pixels (length of 1 means use same box for all classes)
        :param border: Size of image border in pixels to ignore labels within
        :param df: A Pandas DataFrame containing labelled points for birds in a GeoTiff
        :return: A list of strings in the format "<object-class> <x> <y> <width> <height>" as specified here
        https://pjreddie.com/darknet/yolo/
        """

        # Create an empty list to append label strings to
        labels = []
      
        # Iterate through the DataFrame to determine the labels which fall within the image
        for r in df.itertuples():
            if r.x > (min_x+border) and r.x < (min_x+img_size-border) and r.y > (min_y+border) and r.y < (min_y+img_size-border):
                
                try:
                    cls = classes.index(r.LAYER)
                except:
                    continue
                
                # Get box size
                if len(box_sizes) == 1:
                    # Same box size for all labels
                    box_sz = box_sizes[0]
                else:
                    # Specific box size for class
                    box_sz = box_sizes[cls]

                cx = (r.x-min_x)/img_size
                cy = (r.y-min_y)/img_size
                w = box_sz/img_size
                h = box_sz/img_size

                write_str = "%d %0.4f %0.4f %0.4f %0.4f"%(cls,cx,cy,w,h)
                labels.append(write_str)
                
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

            # Calculate tiff width and height in meters
            tiff_width_m = full_tiff.width * full_tiff.res[0]
            tiff_height_m = full_tiff.height * full_tiff.res[1]
            
            # Calculate box size in pixels
            box_sz_px = []
            for box in self.box_sz_m:
                box_sz_px.append(floor(box / full_tiff.res[0]))

            # Calculate border size in pixels
            border_sz_px = floor(self.border_m / full_tiff.res[0])
             
            # Calculate itersize, normalise to either target size in meters or number of pixels
            if self.norm_img:
                # Calculate image size in meters based on overlap and target size in meters
                iter_sz_m = self.img_sz_m - (self.overlap_m * 2)
                
                # Info
                print("Targeting image meter normalised size: {:.2f}m".format(self.img_sz_m))

            else:
                # Update image size in meters (now driven dimension)
                self.img_sz_m = self.target_px * full_tiff.res[0]
                
                # Calculate image size in m based on overlap and target_px
                iter_sz_m = self.img_sz_m - (self.overlap_m * 2)
                
                # Info
                print("Targeting image pixel normalised size: {}px".format(self.target_px))
            
            # Calculate number of columns and rows based on iter_size
            columns = floor(tiff_width_m / iter_sz_m)
            rows = floor(tiff_height_m / iter_sz_m)
            
            # Calculate the iter size in pixels
            iter_sz_px = floor(iter_sz_m / full_tiff.res[0])

            # Calculate image size in pixels
            img_sz_px = floor(self.img_sz_m / full_tiff.res[0])

            # Check for loss of information
            if img_sz_px > self.target_px:
                print("WARNING: Selected frame size {}m results in downscaling to meet target {}x{}px!!!\n".format(self.img_sz_m, self.target_px, self.target_px))
            
            # Print info
            print("TIFF dimensions: {:.2f}m x {:.2f}m".format(tiff_width_m, tiff_height_m))
            print("Image RoI Size: {:.2f}m = {}px".format(iter_sz_m, iter_sz_px))
            print("Image Size (with {:.2f}m overlap): {:.2f}m = {}px".format(self.overlap_m, self.img_sz_m, img_sz_px))
            print("Sectors: {} rows * {} columns = {}\n".format(rows, columns, rows*columns))
            
            # Identify the affine transformation between pixel (x, y) coordinates and latitude and longitude
            fwd = full_tiff.transform
            # Reverse the transformation so latitude and longitude can be converted to pixel (x, y) coordinates
            rev = ~fwd

            # Check ESPG code of orthoimage is the expected 6933 (required for altitude norm)
            crs = int(full_tiff.crs.to_string()[5:])
            if crs != 6933:
                print("ERROR: Expected ESPG6933, try '$ gdalwarp -t_srs EPSG:6933 -co compress=lzw -wo OPTEMIZE_SIZE=YES in.tif out.tif'. Exiting...\n")
                sys.exit()

            # For each bird:
            print("Finding birbs...", end="")
            birds = self.image_coords(self.birds, rev, self.crs_transform)
            print(" Birds found!\n")

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
                print("Slicin yo: {}/{} Done     \r".format(u, columns), end="")
                for v in range(rows):

                    # Calculate (top left?) corner of region
                    x_min = int(u*iter_sz_px)
                    y_min = int(v*iter_sz_px)

                    # Generate label for region
                    label = self.bird_labels(x_min, y_min, img_sz_px, box_sz_px, border_sz_px, birds, self.birds_of_interest)

                    # Get rgb channels fromt tiff
                    # Different memory locations for each channel requires individual read, else resulting array
                    # will not be C-Contiguous (requirement of opencv functions)
                    img_r = full_tiff.read(1, window=Window(x_min, y_min, img_sz_px, img_sz_px))
                    img_g = full_tiff.read(2, window=Window(x_min, y_min, img_sz_px, img_sz_px))
                    img_b = full_tiff.read(3, window=Window(x_min, y_min, img_sz_px, img_sz_px))
                    img = np.dstack((img_b, img_g, img_r))
                    
                    # number of pixels that are all zeros (out of scope)
                    zero_pix = (sum(img_r+img_g+img_b) == 0).sum()

                    # if the image is all black (out of scope) then skip
                    if zero_pix > int(img_sz_px*self.black_thresh):
                        continue
                    
                    # Resize image to target output size if required
                    if self.norm_img:
                        img = cv2.resize(img, (self.target_px, self.target_px), interpolation=cv2.INTER_LANCZOS4)

                    # Generate output names
                    idx_name = 'img-%d-%d'%(u,v)
                    img_wlabel_name = os.path.join(self.folder, 'images_labelled', "{}_{}.png".format(self.output_name, idx_name))
                    img_name = os.path.join(self.folder, 'images', "{}_{}.png".format(self.output_name, idx_name))
                    txt_name = os.path.join(self.folder, 'labels', "{}_{}.txt".format(self.output_name, idx_name))
                    
                    # Write the image
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
                    
                    # Show any images with labels
                    if self.SHOW:
                        
                        # If image has labels
                        if len(label)>0:

                            # For each label
                            for l in label:
                                
                                # Denormalise valuesn from label
                                cls,cx,cy,w,h = l.split(' ')
                                cls = int(cls)
                                cx = float(cx) * self.target_px
                                cy = float(cy) * self.target_px
                                w = float(w) * self.target_px
                                h = float(h) * self.target_px
                                                                
                                # Convert bounding box to cv format
                                top_left = (int(cx-w/2), int(cy-h/2))
                                bottom_right = (int(cx+w/2), int(cy+h/2))
                                
                                # Draw box
                                cv2.rectangle(img, top_left, bottom_right, self.colours[cls], 2)
                                
                                # Draw label
                                cv2.putText(img, str(cls), top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, self.colours[cls], 2)

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
