#!/usr/bin/env python3

"""
Author: Matthew Tavatgis
Created: 2 Feb 2024

=== Birbs ===
Applies augmentations to underepresented classes to increase support
"""

import os
import sys
import cv2
import numpy as np
import imutils
import shutil
import argparse
import random

def arg_parse():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Class balancing data augmentation')

    parser.add_argument("--src", dest = "src_dir",
            help = "Source directory to parse", default = None, type = str)
    parser.add_argument("--name", dest = "out_name",
            help = "Name of new dataset", default = None, type = str)
    parser.add_argument("--src-sfx", dest = "src_sfx",
        help = "Suffix on 'images' / 'labels' folders to use as source", default = None, type = str)

    return parser.parse_args()

def main():
    # Load args
    args = arg_parse()

    # Check source images exist
    image_source_dir = os.path.join(args.src_dir, "images" + (("_" + args.src_sfx) if args.src_sfx is not None else ""))
    if os.path.exists(image_source_dir) == False:
        raise FileNotFoundError("Source directory {} does not exist".format(image_source_dir))

    # Check source labels exist
    label_source_dir = os.path.join(args.src_dir, "labels" + (("_" + args.src_sfx) if args.src_sfx is not None else ""))
    if os.path.exists(image_source_dir) == False:
        raise FileNotFoundError("Source directory {} does not exist".format(label_source_dir))

    # Create ouput directory paths
    image_out_dir = os.path.join(args.src_dir, "images_{}".format(args.out_name))
    label_out_dir = os.path.join(args.src_dir, "labels_{}".format(args.out_name))

    print("\n#################### Creating output directories ####################")
    print("IMAGES: {}\nLABLES: {}\n".format(image_out_dir, label_out_dir))

    # Prompt to overwrite if output dir exists, otherwise attempt to create it
    if os.path.exists(image_out_dir) or os.path.exists(label_out_dir):
        print("WARNING: Output Directory Exists, data will be overwritten ", end="")

        if input("Y/N?:").lower() != "y":
            print("EXITING...\n")
            exit()
        else:
            print("CONTINUING...\n")
            # Delete existing output dir
            try:
                shutil.rmtree(image_out_dir)
                shutil.rmtree(label_out_dir)
            except OSError as error:
                print(error)
                sys.exit()

    # Try to create output directories
    try:
        os.mkdir(image_out_dir)
        os.mkdir(label_out_dir)
    except OSError as error:
        print(error)
        sys.exit()

    # Search source folders for matching image and label
    src_image_names = os.listdir(image_source_dir)
    src_label_names = os.listdir(label_source_dir)

    # Init log of missing entries
    missing = 0
    success = 0

    # Info
    print("######################## Processing Entries #########################")
    print("{} Entries found...".format(len(src_image_names)))
    
    class_counts = []
    labelled_images = []
    for i, name in enumerate(src_image_names):

        # Get identifier
        ident = name.split('.')[0]

        # Get image type
        img_ext = name.split('.')[-1]

        # Create expected label path
        src_image_path = os.path.join(image_source_dir, "{}.{}".format(ident, img_ext))
        src_label_path = os.path.join(label_source_dir, "{}.txt".format(ident))

        # Check expected image label exists
        if os.path.exists(src_label_path) == False:
            print("Warning: Label file {} not found, skipping".format(src_label_path))
            missing = missing + 1
            continue
        
        # Copy original image and label to new output
        dst_image_path = os.path.join(image_out_dir, "{}.{}".format(ident, img_ext))
        dst_label_path = os.path.join(label_out_dir, "{}.txt".format(ident))
        #copy_file(src_image_path, dst_image_path)
        #copy_file(src_label_path, dst_label_path)
        
        # Load label file, get boxes
        class_ids = [] 
        boxes = []
        with open(src_label_path, 'r') as label_file:
            
            # Read first line and extract box
            labels = label_file.readlines()
            if len(labels) != 0:
                
                # Iterate over labels
                for label in labels:
                    
                    # Split line
                    label = label.strip()
                    class_id = label.split(" ")[0]
                    box = label.split(" ")[-4:]
                    box = list(map(float, box))

                    # Convert box from [x_cent, y_cent, width, height] to [x_min, y_min, x_max, y_max]
                    x_min = box[0] - (box[2] / 2)
                    x_max = box[0] + (box[2] / 2)
                    y_min = box[1] - (box[3] / 2)
                    y_max = box[1] + (box[3] / 2)
                    
                    # Save outputs
                    class_ids.append(class_id)
                    boxes.append([x_min, y_min, x_max, y_max])
        
        # Close file
        label_file.close()
        
        # Update class stats
        class_counts += class_ids

        # Save details as a object
        labelled_images.append(LabelledImage(ident, dst_image_path, class_ids, boxes))

    # Determine type and number of augs based on classes present
    class_names, n_instances = np.unique(class_counts, return_counts=True)
    max_count = max(n_instances)
    target_n_aug = []
    for count in n_instances: target_n_aug.append(max_count - count)
    class_stats = dict(zip(class_names, zip(n_instances, target_n_aug)))
    
    # Need to avoid duplicating images with very high counts of common classes


    """
    # Create new names and output paths
    dst_image_paths = []
    dst_label_paths = []
    for aug_ident in len(augs):
        dst_image_paths.append(os.path.join(image_out_dir, "{}.{}".format(aug_ident, img_ext)))
        dst_label_paths.append(os.path.join(label_out_dir, "{}.txt".format(aug_ident)))
    """
    
    # Run augmentations

class LabelledImage():
    def __init__(self, ident, img_path, class_ids, boxes):
        # Save vars
        self.ident = ident
        self.img_path = img_path
        self.class_ids = class_ids
        self.boxes = boxes
 
def rotate_aug(src_image, class_ids, boxes, rotations):
    """
    Create rotated coppies of a source image and corresponding label files
    param src_image: image to operate on
    param class_ids: list of classes present
    param boxes: list of boxes present in yolo format [[x_center, y_center, width, height], ...] (normalised)
    param rotations: list of rotations to implement (in degrees)
    return: [images], [labels]
    """

    images = []
    labels = []
    for angle in rotations:
        
        # Create rotated image
        img_rotated = imutils.rotate_bound(source_image, rot_opt[angle])

        # Write rotated image to output directory
        cv2.imwrite(dst_image_path, img_rotated)
        
        # Apply augmentations to labels found
        rotate_org(class_ids, boxes)
    
        # Rotate bounding box
        yolo_labels = ""
        for box in boxes:
            
            # Create a fake image and draw the box in then rotate lol
            box_img = np.zeros([source_image.shape[0], source_image.shape[1], 1], dtype=np.uint8)
            box_img = draw_frame(box_img, box)
            box_img_rot = imutils.rotate_bound(box_img, rot_opt[angle])

            # Extract new box from rotated image
            x_vals = np.nonzero(np.argmax(box_img_rot, axis=0))[0]
            y_vals = np.nonzero(np.argmax(box_img_rot, axis=1))[0]
            x_max = max(x_vals)
            x_min = min(x_vals)
            y_max = max(y_vals)
            y_min = min(y_vals)
            
            # Caculate yolo box
            yolo_boxes.append(yolo_box(image_rotated.shape, box))

            # Draw box
            #cv2.rectangle(img_rotated, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            #cv2.imshow("test", img_rotated)
            #cv2.imshow("why", box_img_rot)
            #if cv2.waitKey(0) == ord('q'):
            #    cv2.destroyAllWindows()
            #    sys.exit()
            
            # Populate data
            yolo_lables += "{} {} {} {} {}\n".format(class_id, x_center, y_center, width, height)

def copy_file(src_dir, dst_dir):
    """ 
    Helper function to safely copy files
    """
    try:
        shutil.copy(src_dir, dst_dir)
    except OSError as error:
        print(error)
        print("\nFailed to copy {} to {}".format(src_dir, dst_dir))
        sys.exit()

def yolo_box(image_shape, box):
    """
    Helper function to create yolo boxes
    param image_shape: [rows, columns, ...], image shape
    param box: bounding box in pixel coords [x_min, y_min, x_max, y_max]
    return: bounding box normalised to image size [x_center, y_center, width, height]
    """

    # Normalise values
    x_min = box[0] / image_shape[1]
    y_min = box[1] / image_shape[0]
    x_max = box[2] / image_shape[1]
    y_max = box[3] / image_shape[0]
    
    # Convert to yolo format
    width = x_max - x_min
    height = y_max - y_min
    x_center = x_min + (width / 2)
    y_center = y_min + (height / 2)
    
    # Return in yolo format
    return [x_center, y_center, width, height]

def draw_frame(frame, box):
    """
    Draw yolo format bounding box on image and return result
    param frame: image to operate on
    param box: yolo bounding box, [x_center, y_center, width, height] (normalised)
    return: copy of image with box drawn
    """
    # Make a copy to avoid overwriting source
    ret = frame.copy()

    # Normalise box values to image shape
    normVals = np.full(len(box), ret.shape[0])
    normVals[::2] = ret.shape[1]
    box = (np.clip(np.array(box), 0, 1) * normVals).astype(int)
    
    # Draw box
    cv2.rectangle(ret, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

    return ret

if __name__ == '__main__':
    main()
