#!/usr/bin/env python3

"""
Author: Matthew Tavatgis
Created: 16 Jan 2024

=== Birbs ===
Script to look through tif labeling CSVs, search for a certain label and replace all instance with a desired new label. Writes modified files to a new folder "blitz", leaving originals intact. By default iterates on already output files, use --ingest flag to bring in fresh coppies
"""

import os
import argparse
import pandas as pd
from distutils.util import strtobool

def arg_parse():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Label modifer for birds!')

    parser.add_argument("--target", dest = "label_target",
            help = "Label to find and replace", default = None, type = str)
    parser.add_argument("--new", dest = "new_label",
            help = "What to replace the target label with", default = None, type = str)
    parser.add_argument("--ingest", dest = "ingest_data",
            help = "Optionally use pwd as source rather than expected output", default = False, type = lambda x: bool(strtobool(x))) 
    
    return parser.parse_args()

def main():
    
    # Welcome
    print("\nWELCOME TO BLITZ LABEL HUNTER!\n")

    # Read args
    args = arg_parse()
    
    # Working directory
    pwd = os.getcwd()

    # Create output path
    out_path = os.path.join(pwd, "blitz")
    if os.path.exists(out_path) == False and args.ingest_data == False:
        raise RuntimeError("ERROR, no exisiting outputs, run with '--ingest True' to bring in new data from pwd")
        exit()

    # Find csv files
    if args.ingest_data:
        src_dir = pwd
        print("INGESTING FRESH DATA\n")
    else:
        src_dir = out_path
    
    files = os.listdir(src_dir)

    print("SEARCHING '{}', FOUND:".format(src_dir))
    csv_files = []
    for f in files:
        if f.endswith('.csv'):
            csv_files.append(f)
            print(f)
    
    # Create summary of changes
    target_count = []
    print("\nCount:")
    for f in csv_files:
        # Input / Output
        file_in = os.path.join(src_dir, f)
        
        # Read in data
        data_in = pd.read_csv(file_in)
        
        # Count instances (Pandas Sucks!!!)
        try:
            target_count.append(data_in['LAYER'].value_counts()[args.label_target])
        except:
            target_count.append(0)
        print(target_count)

    # Confirm operation
    print("\nEDITING {} LABLES: '{}' -> '{}', CONTINUE ".format(sum(target_count), args.label_target, args.new_label), end="")
    if input("Y/N?: ").lower() != "y":
        print("EXITING...\n")
        exit()
    else:
        print("CONTINUING...\n")

    # Create output dir
    if os.path.exists(out_path):
        print("WARNING: Output file exists, data will be overwritten ", end="")

        if input("Y/N?: ").lower() != "y":
            print("EXITING...\n")
            exit()
        else:
            print("CONTINUING...\n")
    else:
        os.mkdir(out_path)

    # Modify files
    for f in csv_files:
        # Input / Output
        file_in = os.path.join(src_dir, f)
        file_out = os.path.join(out_path, f)
        
        # Read in data
        data_in = pd.read_csv(file_in)
        
        # Go hunting
        data_in['LAYER'].replace(args.label_target, args.new_label, inplace = True)
        
        # Save result
        data_in.to_csv(file_out, index=False)

    # Done
    print("OPERATION COMPLETE! REPLACED {} INSTANCES OF '{}' with '{}'\n".format(sum(target_count), args.label_target, args.new_label))

if __name__ == "__main__":
    main()
