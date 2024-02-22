#!/usr/bin/env python3

"""
Author: Matthew Tavatgis
Created: 27 Jan 2024

=== Birbs ===
Use a common repository of raw datasets to dynamically (and repeatedly) recreate a meta dataset for use with training.
Alleviates disk space requirements of storing all meta datasets as their own entinity.
Generates a record file that keeps track of parameters used for each meta dataset

TODO:
    - Save a copy of the record file to git repo?
"""

import os
import sys
import pwd
import yaml
import shutil
import datetime
import argparse
import pandas as pd
from natsort import natsorted, ns
from distutils.util import strtobool

def arg_parse():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Generate Meta Dataset')

    parser.add_argument("--src", dest = "src",
            help = "List of raw datasets to source", nargs = '*', default = None, type = str)
    parser.add_argument("--name", dest = "name",
            help = "Name to call metadataset", default = None, type = str)
    parser.add_argument("--labels", dest = "label_def",
            help = "Label definition yaml file", default = None, type = str)

    return parser.parse_args()

def main():

    # Welcome
    print("\nWELCOME TO DATA SMASH!\n")

    # Read args
    args = arg_parse()
    
    # Sort input paths for consistent parsing across runs
    args.src = natsorted(args.src)

    # Validate source directories are of execpted structure, generate list of dicts describing each source
    src_cfgs = []
    for src_dir in args.src:
        # Create list of validated paths for each source 
        src_cfgs.append(validate_src_dir(src_dir))

    # Create output directory, clean if it exists
    cwd = os.getcwd()
    out_paths = create_output(cwd, args.name, allow_delete=True)
    
    """ :) lol yolo why
    # Read label config file
    if not os.path.exists(args.label_def):
        print("ERROR: Label definition '{}' does not exist! EXITING...\n".format(args.label_def))
        sys.exit()

    label_cfg = None
    with open(args.label_def, "r") as stream:
        label_cfg = yaml.safe_load(stream)
    
    # Check output is valid
    if label_cfg is None:
        print("ERROR: Failed to read label file! Check format! EXITING...\n")
        sys.exit()
    
    # Add label convention of each source to src_cfg
    classes = []
    for dataset in src_cfgs:
        
        # Compare labels to offical config list
        with open(dataset['Label File'], "r") as stream:
            lines = stream.readlines()
            
            # Check if item exists in config list, write label mapping to src_cfg
            for i, label in enumerate(lines):

                # Strip line feed of label name, add to list
                label = label.rstrip()
                classes.append(label)

                # Search for label name in label_cfg index
                try:
                    idx = list(label_cfg.keys()).index(label)
                    dataset[str(i)] = str(idx)
                except ValueError:
                    print("ERROR: Label '{}' from '{}' does not exist in '{}'! EXITING...\n".format(label, dataset['source'], args.label_def))
                    sys.exit()

    # Reduce classes to unique entries
    classes = sorted([*{*classes}])
    """
    
    # Get list of all classes present
    classes = []
    for dataset in src_cfgs:
        
        with open(dataset['Label File'], "r") as stream:
            lines = stream.readlines()
            for label in lines: classes.append(label.rstrip())

    # Reduce classes to unique entries
    classes = sorted([*{*classes}])
    
    # Add mapping to full class list to each sources' config
    for dataset in src_cfgs:
        
        with open(dataset['Label File'], "r") as stream:
            lines = stream.readlines()
            
            for i, label in enumerate(lines):
                label = label.rstrip()
            
                # Search for label name in classes index, write mapping to source config
                try:
                    idx = classes.index(label)
                    dataset[str(i)] = str(idx)
                except ValueError as error:
                    print(error)
                    sys.exit()
    
    # Iterate over label files from each source, create a copy with rectified class id
    count = 0
    source_counts = []
    for source in src_cfgs:
        
        # Read files
        labels_files = natsorted(os.listdir(source['Labels']))
        images_files = natsorted(os.listdir(source['Images']))
        images_labelled_files = natsorted(os.listdir(source['Images Labelled']))
        
        # Check numbers match
        if (len(images_files) + len(images_labelled_files)) != len(labels_files):
            print("ERROR: Total number of images does not equal total number of labels in {}! EXITING...\n".format(source['Source Root']))
            sys.exit()

        # Iterate over label files, check label is txt, look for corresponding image
        for label in labels_files:
            
            # Get original name, form new name combining MetaSet name, index and old name
            name = label.split('/')[-1]
            name = name.split('.')[0]
            new_name = "{}_{}_{}".format(count, args.name, name)
            new_label = "{}.txt".format(new_name)
            new_image = "{}.png".format(new_name)
            new_label = os.path.join(out_paths['Labels'], new_label)
            new_image = os.path.join(out_paths['Images'], new_image)
            
            # Check corresponding image exists
            src_img_path_a = os.path.join(source['Images'], "{}.png".format(name))
            src_img_path_b = os.path.join(source['Images Labelled'], "{}.png".format(name))
            if os.path.exists(src_img_path_a):
                src_img_path = src_img_path_a
            elif os.path.exists(src_img_path_b):
                src_img_path = src_img_path_b
            else:
                print("ERROR: Corresponding image not found for label '{}'! EXITING...\n".format(label))
                sys.exit()

            # Read exisiting file
            old_lines = []
            with open(os.path.join(source['Labels'], label), "r") as stream:
                old_lines = stream.readlines()

            # For each line replace class label then write to output
            with open(new_label, "w") as stream:
                for line in old_lines:
                    new_line = line.replace(line[0], source[str(line[0])], 1) 
                    stream.write(new_line)
            
            # Copy image
            copy_file(src_img_path, new_image)

            # Update iterator
            count = count + 1

        # Update source count
        source_counts.append(count - sum(source_counts[:i]))
    
    # Write yolo data file
    # Form data dict:
    data = {}
    with open(out_paths['Yolo File'], "w") as stream:

        data['path'] = out_paths['MetaSet']
        data['train'] = "train/images"
        data['val'] = "valid/images"
        
        # Save class names
        names = {}
        for i, class_name in enumerate(classes):
            names[i] = class_name
        data['names'] = names

        """ Sadness :(
        # Get list of class label ids from official label_config
        idx = []
        class_names = []
        for class_name in classes:
            idx.append(list(label_cfg.keys()).index(class_name))
            class_names.append(class_name)
        
        # Sort class names according to id and populate label names in file
        data['names'] = dict(sorted(zip(idx, class_names)))
        """

        # Dump to yaml
        yaml.dump(data, stream, default_flow_style=False, sort_keys=False)
    
    # Write yolo test file
    with open(out_paths['Yolo Test'], "w") as stream:
        
        # Edit validation path
        data['val'] = "test/images"

        # Dump to yaml
        yaml.dump(data, stream, default_flow_style=False, sort_keys=False)

    # Write info to record file
    with open(out_paths['Record File'], "w") as stream:
        
        # Welcome
        stream.write("Data Smash Record File For {}!\n\n".format(args.name))
        
        # Time/date/user
        stream.write("Date: {}\n".format(datetime.datetime.now()))
        stream.write("User: {}\n\n".format(pwd.getpwuid(os.getuid())[0]))

        # Command
        stream.write("Invoked with:\n")
        stream.write("python3 data_smash.py --src ")
        for source in args.src:
            stream.write("{} ".format(source))
        stream.write("--name {} ".format(args.name))
        stream.write("--labels {}\n\n".format(args.label_def))

        # Stats
        stream.write("Outputs include:\n")
        for i, source in enumerate(src_cfgs):
            stream.write("{}: {} image label pairs\n".format(source['Source Root'], source_counts[i]))

        # Label mapping
        stream.write("\nOutput label mapping:\n")
        for i, class_name in enumerate(classes):
            stream.write("{}: {}\n".format(i, class_name))

    # Info
    print("Doneski.")

def copy_file(src_dir, dst_dir):
    """ Helper function for file coppies"""
    try:
        shutil.copy(src_dir, dst_dir)
    except OSError as error:
        print(error)
        print("\nFailed to copy {} to {}".format(src_dir, dst_dir))
        sys.exit()

def create_output(out_path, name, allow_delete = False):
    """
    Check if a given output directory exists
    If 'allow_delete' = True give the user the option to overwrite an existing MetaSet
    Finally, create output directory structure:
    \out_path
        \SmashSets                  - Root path
            \config_records         - Perpetual storage of metaset run details
                \<name>_record.txt  - Record file describing this run
            \MetaSet                - Meta Data set root
                \images_train       - images to train on
                \labels_train       - lables to train on
                \data.yaml          - yolo data file
                \test.yaml          - yolo test file
    """
    # Define Keys
    keys = ['SmashRoot', 'MetaSet', 'Images', 'Labels', 'Records']
    
    # Define dirs
    dirs = [os.path.join(out_path, "SmashSets"),
            os.path.join(out_path, "SmashSets", "MetaSet"),
            os.path.join(out_path, "SmashSets", "MetaSet", "images_train"),
            os.path.join(out_path, "SmashSets", "MetaSet", "labels_train"),
            os.path.join(out_path, "SmashSets", "config_records")]
    
    # Create dict
    out_dict = dict(zip(keys, dirs))
    
    # Check if record file exists, create if it doesnt
    record_file_path = os.path.join(out_dict['Records'], "{}_record.txt".format(name))
    if os.path.exists(record_file_path):
        print("ERROR: Record file '{}_record.txt' already in {}! EXITING...\n".format(name, out_dict['SmashRoot']))
        sys.exit()
    else:
        # Create record file
        with open(record_file_path, "w") as stream:
            pass

    # Check if meta set path exists
    if os.path.exists(out_dict['MetaSet']):
        
        # Exit if deletion illegal
        if allow_delete == False:
            print("ERORR: Output directory '{}' exists, deletion set as illegal. EXITING...\n".format(out_path))
            sys.exit()
        
        # Warn if overwrite is legal
        resp = "WARNING: Output directory '{}' exists, data will be overwritten! Y/N?: ".format(out_dict['MetaSet'])
        if input(resp).lower() != "y":
            
            # First remove defunct record file
            try:
                os.remove(record_file_path)
            except OSError as error:
                print(error)
                sys.exit()

            print("EXITING...\n")
            sys.exit()
        
        # Try to delete existing folder
        print("CONTINUING...\n")
        try:
            shutil.rmtree(out_dict['MetaSet'])
        except OSError as error:
            print(error)
            sys.exit()

    # Try to create output directories that dont exist
    for dir_out in out_dict.values():
        if not os.path.exists(dir_out):
            try:
                os.mkdir(dir_out)
            except OSError as error:
                print(error)
                sys.exit()

    # Create yolo data file
    yolo_data_path = os.path.join(out_dict['MetaSet'], "data.yaml")
    with open(yolo_data_path, "w") as stream:
        pass
    
    # Create yolo test file
    yolo_test_path = os.path.join(out_dict['MetaSet'], "test.yaml")
    with open(yolo_test_path, "w") as stream:
        pass
    
    # Add record and yolo files to dict
    out_dict['Record File'] = record_file_path
    out_dict['Yolo File'] = yolo_data_path
    out_dict['Yolo Test'] = yolo_test_path

    # Return output paths
    return out_dict

def validate_src_dir(src_path):
    """ 
    Check that a given source directory is of the format:
    \src_path
        \images             - containing unlablled images (png)
        \images_labelled    - containing labelled images (png)
        \labels             - containing yolo format txt labels
        \label_file         - label description file
    """
    # Define keys
    keys = ['Source Root', 'Images', 'Images Labelled', 'Labels', 'Label File']
    
    # Define dirs
    dirs = [src_path,
            os.path.join(src_path, "images"),
            os.path.join(src_path, "images_labelled"),
            os.path.join(src_path, "labels"),
            os.path.join(src_path, "classes.txt")]
    
    # Create dict
    in_dict = dict(zip(keys, dirs))

    # Check dirs exist
    for folder in in_dict.values():
        if not os.path.exists(folder):
            raise FileNotFoundError("ERROR, source '{}' is invalid, '{}' not found! EXITING...\n".format(src_path, folder))
            sys.exit()
    
    # Retrun validated paths
    return in_dict

if __name__ == '__main__':
    main()
