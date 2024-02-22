#!/usr/bin/env python3

"""
Author: Matthew Tavatgis
Created: 27 Jan 2024

=== Birbs ===
Parses yolo label files to summarise class support
"""

import os
import sys
import argparse
import collections

def arg_parse():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Summarise YOLO labels')

    parser.add_argument("--src", dest = "src",
            help = "Source directory to parse for labels", default = None, type = str)

    return parser.parse_args()

def main():

    # Welcome
    print("\nListing those labels!\n")

    # Read args
    args = arg_parse()

    # Check source directory exists
    if not os.path.exists(args.src):
        print("ERROR: Source dir '{}' does not exist! EXITING...\n".format(args.src))
        sys.exit()

    # Get source contents and iterate
    empty = 0
    classes = []
    files = os.listdir(args.src)
    for f in files:
            
        # Check is txt
        if not f.endswith('.txt'):
            continue
        
        if os.stat(os.path.join(args.src, f)).st_size == 0:
                empty += 1

        # Read file
        with open(os.path.join(args.src, f), "r") as stream:
            lines = stream.readlines()

            for l in lines:
                l = l.split(' ')
                classes.append(int(l[0]))

    # Count classes read
    counts = collections.Counter(classes)

    # Print result
    print(collections.OrderedDict(sorted(counts.items())))

    # Print number of empty
    print("Empty: {}".format(empty))

if __name__ == '__main__':
    main()
