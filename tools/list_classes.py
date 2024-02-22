#!/usr/bin/env python3

"""
Author: Matthew Tavatgis
Created: 16 Jan 2024

=== Birbs ===
Script to look through tif labeling CSVs and display labels of each file, as well as summarise all unique labels
"""

import os
import numpy as np
import pandas as pd
import argparse

def arg_parse():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Class checker for CSV data files')

    parser.add_argument("--src", dest = "src_dir",
            help = "Source directory to search for files", default = "/home/matt/Birds/Processed/CSVs", type = str)

    return parser.parse_args()

def main():

    # Get arguments
    args = arg_parse()

    # Set directory to look in and find files
    root = args.src_dir
    contents = os.listdir(root)
    
    # Find valid files
    csv_files = []
    for f in contents:
        # Check if file is csv
        if f.endswith(('.csv')):
            csv_files.append(f)
    
    if csv_files != []:
        csv_files = sorted(csv_files)
    else:
        print("\nNO VALID FILES FOUND! EXITING...\n")
        exit()

    # Read in files
    data = []
    for f in csv_files:
         
        # Read in data
        table = pd.read_csv(os.path.join(root, f))

        # Append source file name and add to data list
        table.insert(0, 'SRC', f)
        data.append(table)
        print(f)
        print(sorted(data[-1]['LAYER'].unique()))
        print("\n")

    # Combine frame and check unique labels
    combined = pd.concat(data)
    labels = sorted(combined['LAYER'].unique())
    totals = []
    for i in labels:
        print("=== {} ===".format(i))

        # Form subset of data assoicated with these lables
        subset = combined.loc[combined['LAYER'] == i]

        # Get unique sources present
        sources = subset['SRC'].unique()

        # Get counts assoicates with each source
        count = []
        for src in sources:
            count.append(subset['SRC'].value_counts()[src])
            print("{}: {}".format(src, count[-1]))

        print("Total: {}\n".format(sum(count)))
        totals.append(sum(count))

    # Summary print
    print("===== Summary =====")
    for n, i in enumerate(labels):
        print("{}: {}".format(i, totals[n]))
    print("")

if __name__ == '__main__':
    main()
