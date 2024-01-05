# Takes in large geotiff and a csv
# Crops in geotiff in a sliding window manner, saving each image with associated labels from csv in different folders
# Written by: Serena Mou, based on some code from Jaime
# Date: 3 Jan, 2024


import rasterio as rio
import pandas as pd
from osgeo import osr
from math import floor
from rasterio.windows import Window
import numpy as np
import cv2
import os
import random


class CutGeotiff():


    def __init__(self):
            # Define output image size
        self.img_sz = 512
        overlap_percent = 10
        self.overlap = round(self.img_sz*overlap_percent/100)
        self.iter_size = round(self.img_sz-self.overlap)
        # Define box size
        self.box_sz = 50
        self.border = int(0.2*self.box_sz)

        self.SHOW = False
        self.WRITE = True
        self.DEBUG = False


        root = "/home/serena/Data/Birds/Serena/Processed/EF_Data/Trip_1/"

        # Define output folder
        folder_name = 'EFTrip_1_sz%d_overlap%d_box%d_border%d'%(self.img_sz,overlap_percent,self.box_sz,self.border)
        
        self.folder = os.path.join(root, folder_name)
        sub_dirs = ['images', 'labels', 'images_labelled']
        if self.WRITE:
            if not os.path.isdir(self.folder):
                os.mkdir(self.folder)
            
            for dir in sub_dirs:
                full = os.path.join(self.folder, dir)
                if not os.path.isdir(full):
                    os.mkdir(full)



        # Define the file containing bird labels and locations, then read the file into a pandas dataframe
        bird_points = os.path.join(root, 'EF_Trip1_All.csv')
        # bird_points = os.path.join(root, 'raw_data/2018-06-28/Labels/RAW_RaineIsland-BirdCount-MGA55_20180628.csv')
        self.birds = pd.read_csv(bird_points)
        #self.tiff = os.path.join(root,'raw_data/2018-06-28/GeoTIFF/RAW_RaineIsland-Ortho-1cm-rectified-COG_20180628.tif')
        self.tiff = os.path.join(root,'EF_Trip1_Ortho_Export_2.tif')
        
        self.birds_of_interest = []
        
        # change self.birds_of_interest if not all species are required
        self.get_all_species()
        print(self.birds_of_interest)
        # self.birds_of_interest = ["Lesser Frigatebird", "Common Noddy", "Masked Booby", "Red-Footed Booby", "Greater Frigatebird", 
        #                           "Brown Booby", "Black Noddy", "Red-Tailed Tropicbird", "Sooty Tern", "Silver Gull"]
        self.colours = [(0,110,170),(128,128,128),(75,25,230),(128,0,0),(25,230,230),(180,30,145),(60,180,60),(212,190,250)]
        # birds_of_interest = ["Masked Booby", "Red-Footed Booby"]

        # Define the ESPG code for the label coordinates
        self.source = osr.SpatialReference()
        self.source.ImportFromEPSG(28355)
        
        self.black_thresh = 0.5


    def get_all_species(self):
        all_birds = self.birds['LAYER'].unique()
        for bird in all_birds:
            # sort for unwanted species here
            self.birds_of_interest.append(bird)

        # sort alphabetical
        self.birds_of_interest = sorted(self.birds_of_interest)
        return

    def image_coords(self, df, trans, transform):
        """
        A function to determine the (x, y) image coordinates for each labelled bird.
        :param df: A Pandas DataFrame containing labelled points for birds in a GeoTiff
        :param trans: The reverse the transformation to convert latitude and longitude to pixel (x, y) coordinates
        :return: The original Pandas DataFrame with two extra columns, "x" and "y" which contain the x & y image
        coordinates respectively.
        """
        for i, r in df.iterrows():
            # Determine the ortho coordinates
            # Not needed if lat lon are already in correct format
            #lat, lon, _ = transform.TransformPoint(r.POINT_X, r.POINT_Y)
            lat, lon = r.LAT, r.LON
            # Determine the pixel coordinates
            x, y = trans * (lon, lat)
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
                
                cls = classes.index(r.LAYER)
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


        # Open the orthoimage
        print("Opening Orthoimage...")
        with rio.open(self.tiff) as full_tiff:
            
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
            print("Finding birbs...")
            birds = self.image_coords(self.birds, rev, transform)

            if self.WRITE:
                with open(self.folder + "classes.txt", "w") as f:
                    for c in self.birds_of_interest:
                        f.write(c.replace(" ", "") + "\n")
         

            for u in range(columns):
                for v in range(rows):
                    x_min = int(u*self.iter_size)
                    y_min = int(v*self.iter_size)
                    #print(u,v,x_min,y_min)

                    label = self.bird_labels(x_min, y_min, birds, self.birds_of_interest)

                    img_r = full_tiff.read(1, window=Window(x_min, y_min, self.img_sz, self.img_sz))
                    img_g = full_tiff.read(2, window=Window(x_min, y_min, self.img_sz, self.img_sz))
                    img_b = full_tiff.read(3, window=Window(x_min, y_min, self.img_sz, self.img_sz))
                    
                    # if the image is all black (out of scope) then skip
                    # number of pixels that are all zeros (out of scope)
                    
                    zero_pix = (sum(img_r+img_g+img_b) == 0).sum()
                   
                    if zero_pix > int(self.img_sz*self.black_thresh):
                        continue

                    img = np.dstack((img_b, img_g, img_r))
                    
                    idx_name = 'img-%d-%d'%(u,v)
                    img_wlabel_name = os.path.join(self.folder,'images_labelled',idx_name+'.jpg')
                    img_name = os.path.join(self.folder,'images',idx_name+'.jpg')
                    txt_name = os.path.join(self.folder,'labels',idx_name+'.txt')

                    if self.WRITE:
                    
                        if len(label)>0:
                            cv2.imwrite(img_wlabel_name,img)
                            with open(txt_name,'w') as f:
                                for l in label:
                                    f.write("%s\n"%l)
                        else: 
                            cv2.imwrite(img_name, img)
                        

                    if self.SHOW:
                        if len(label)>0:
                            print(label)
                            for l in label:
                                
                                cls,cx,cy,w,h = l.split(' ')
                                cls = int(cls)
                                cx = float(cx)
                                cy = float(cy)
                                top_left = (int(cx*self.img_sz-0.5*self.box_sz),int(cy*self.img_sz-0.5*self.box_sz))
                                bottom_right = (int(cx*self.img_sz+0.5*self.box_sz),int(cy*self.img_sz+0.5*self.box_sz))
                                color = self.colours[cls]
                                # print(top_left, bottom_right, color)
                                cv2.rectangle(img,top_left, bottom_right, color, 5)
                            
                            cv2.imshow("named",img)
                            cv2.waitKey(0)
                            input()
                            cv2.destroyAllWindows()


        # Close the GeoTiff file
        full_tiff.close()

        # Print statement to signify that the script has finished running
        print("Birb extraction complete!")


def main():
    cg = CutGeotiff()
    cg.run()

if __name__ == '__main__':
    main()
