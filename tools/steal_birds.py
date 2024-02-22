#!/usr/bin/env python3

"""
Author: Matthew Tavatgis
Created: 9th Feb 2024

=== Birbs ===
Extracts instances of rare bird classes and places them in exisiting images to increase training support
Preferetially selects images with the least birds in them
Applies blur to blend images

TODO
 - Make augmentations reloop on target images if aug_target isnt met in a single pass
 - Information, statistics, prints
 - Diversify augmentation ident on image_index loop
"""
import os
import sys
import cv2
import copy
import math
import random
import imutils
import argparse
import imagesize
import numpy as np
from natsort import natsorted, ns

def arg_parse():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Rare Bird Stealer')

    parser.add_argument("--src", dest = "src",
            help = "List of raw datasets to source", default = None, type = str)
    
    return parser.parse_args()

def main():

    # Welcome
    print("\nWELCOME TO BIRD STEALER!\n")
    
    # Config
    delta_tol = 0.6             # Skip augmenting classes within delta_tol percent of top class
    target_pct = 0.4            # Percentage of max class support to target for classes being augmented
    thief_inflate = 0.4         # Inflation percentage to take more image context with stolen bird
    rotations = [90, 180, 270]  # Rotations to implement for augmentation (must be integer mult 90)
    mirror_axis = [0, 1]        # Axis to mirror on for augmentation (must be 0-1)
    max_attempts = 100          # Number of tries randomly placing birds in each target image
    collision_tol = 0.4         # Percentage buffer around placed bird to inflate collision detection
    blend_inner = 0.2           # Percentage of placed bird size to expand into bounding box when blending
    blend_outer = 0.4           # Percentage of placed bird size to expand away from bounding box when blending (<= collision_tol)

    # Read args
    args = arg_parse()

    # Validate source directory
    src_cfg = validate_src_dir(args.src)

    # Warn about file modification
    prompt = "WARNING: Procedure will add files to original source directory! Proceed? Y/N: "
    if input(prompt).lower() == "y":
        print("CONTINUING...\n")
    else:
        print("EXITING...\n")
        sys.exit()
    
    # Search source folders for matching image and label
    src_image_names = natsorted(os.listdir(src_cfg['Images']))
    src_label_names = natsorted(os.listdir(src_cfg['Labels']))
    
    # Purge prexisting outputs from bird stealer
    prompt = "WARNING: Exisitng bird stealer outputs (*_aug-*.txt, *_aug-*.png) will be purged!!! Proceed? Y/N: "
    if input(prompt).lower() == "y":
        print("CONTINUING...\n")
    else:
        print("EXITING...\n")
        sys.exit()
    
    purge_image_names = copy.copy(src_image_names)
    for image_path in src_image_names:
        if "aug" in image_path:
            tgt_path = os.path.join(src_cfg['Images'], image_path)
            try:
                os.remove(tgt_path)
                purge_image_names.remove(image_path)
            except:
                print("WARNING: Failed to remove pre-existing image {}!!!".format(tgt_path))

    purge_label_names = copy.copy(src_label_names)
    for label_path in src_label_names:
        if "_aug-" in label_path:
            tgt_path = os.path.join(src_cfg['Labels'], label_path)
            try:
                os.remove(tgt_path)
                purge_label_names.remove(label_path)
            except:
                print("WARNING: Failed to remove pre-existing label {}!!!".format(tgt_path))
                continue

    # Init log of missing entries
    missing = 0
    success = 0

    # Info
    print("######################## Processing Entries #########################")
    print("{} Entries found...".format(len(purge_image_names)))

    # Collect label info
    label_index = {}
    class_counts = []
    unique_classes = []
    labelled_images = []
    for i, name in enumerate(purge_image_names):
        
        # Get identifier
        ident = name.split('.')[0]

        # Get image type
        img_ext = name.split('.')[-1]

        # Create expected label path
        src_image_path = os.path.join(src_cfg['Images'], "{}.{}".format(ident, img_ext))
        src_label_path = os.path.join(src_cfg['Labels'], "{}.txt".format(ident))
        
        # Check expected image label exists
        if os.path.exists(src_label_path) == False:
            print("Warning: Label file {} not found, skipping".format(src_label_path))
            missing = missing + 1
            continue

        # Get image size
        width, height = imagesize.get(src_image_path)

        # Load label file, get boxes
        class_ids = []
        boxes = []
        with open(src_label_path, 'r') as label_file:

            # Read first line and extract box
            labels = label_file.readlines()

            # Iterate over labels
            for label in labels:

                # Split line
                label = label.strip()
                class_id = label.split(" ")[0]
                box = label.split(" ")[-4:]
                box = list(map(float, box))

                # Update unique class entries and expand label index if new class found
                if class_id not in unique_classes:
                    unique_classes.append(class_id)
                    label_index[str(class_id)] = []

                # Save outputs
                class_ids.append(class_id)
                boxes.append(cv_box([width, height], box))
        
        # Close file
        label_file.close()

        # Update label index for each class present
        for bird in unique_classes:
            if bird in class_ids:
                # Add to index and continue to next class to avoid duplicate entries
                label_index[str(bird)].append(ident)
                continue
        
        # Update class stats
        class_counts += class_ids

        # Save details as a object
        labelled_images.append(LabelledImage(ident=ident,
                                        img_path=src_image_path,
                                        img_size=[width, height],
                                        class_ids=class_ids,
                                        boxes=boxes,
                                        n_inst=len(boxes))
        )
    
    # Sort labelled images by number of labels present (least to most), create dictionary from idents
    labelled_images = sorted(labelled_images, key = lambda x: x.n_inst)
    ident_keys = []
    for datum in labelled_images: ident_keys.append(datum.ident)
    image_dict = dict(zip(ident_keys, labelled_images))

    # Determine number of classes present
    class_names, n_instances = np.unique(class_counts, return_counts=True)
    max_count = max(n_instances)
    
    # Determine number to instances to generate for each class
    target_n_augs = []
    for count in n_instances:

        # Dont set target if class is within 20% of top class
        delta = max_count - count
        if delta > (max_count * delta_tol):
            target_n_augs.append(int(delta * target_pct))
        else:
            target_n_augs.append(0)
    
    # Save stats as dictionary
    class_stats = dict(zip(class_names, zip(n_instances, target_n_augs)))
    print(class_stats)

    # Iterate over classes
    for bird_class in class_names:
        
        # Skip class if aug desire is 0
        if class_stats[bird_class][1] == 0: continue

        # Create pool of stolen birds up to desired augmentation size, load all instances
        thieved_birds = []
        for ident in label_index[str(bird_class)]:
            
            # Load target image
            src_img_path = image_dict[ident].img_path
            src_img = cv2.imread(src_img_path)

            # Iterate over boxes, Steal dem birbs (cv box = [x_min, y_min, x_max, y_max])
            for i, bird in enumerate(image_dict[ident].boxes):
                
                if (len(image_dict[ident].boxes) != len(image_dict[ident].class_ids)):
                    sys.exit()

                # Check class matches current search
                if image_dict[ident].class_ids[i] == bird_class:

                    # Inflate bounding box to get more image context
                    bird_box = inflate_box(bird, [bird[3]-bird[1], bird[2]-bird[0]], image_dict[ident].img_size, thief_inflate)

                    # Get list of boxes minus current box
                    cand_boxes = copy.copy(image_dict[ident].boxes)
                    del cand_boxes[i]
                    
                    # Check for collision, ensure box doesnt include other birds
                    if check_collision(bird_box, cand_boxes) == False:

                        # No collisions, append result
                        thieved_birds.append(src_img[bird_box[1]:bird_box[3], bird_box[0]:bird_box[2]])
        
        # Augment thieved birds to generate greater pool
        bird_clones = []
        for stolen_chicken in thieved_birds:
            
            # Create list of aug birds
            aug_birds = [stolen_chicken]
            
            # Create base augmentations - rotate 90, 180, 270 degrees 
            rotated_birds, _ = rotate_aug(aug_birds[0], [], rotations)
            aug_birds += rotated_birds

            # Mirror all 4 rotations on both x and y
            for i in range(len(aug_birds)):
                mirrored_birds, _ = mirror_aug(aug_birds[i], [], mirror_axis)
                aug_birds += mirrored_birds
            
            # Add results to main output
            bird_clones += aug_birds
        
        # Randomly shuffle birdclones so that augments are not placed together
        random.shuffle(bird_clones)

        # Check number of birds found
        n_stolen = len(bird_clones)
        print("\nFound {} birds to steal for class '{}', augmented to {} total!".format(len(thieved_birds), bird_class, n_stolen))

        # Begin augmentation loop, while desired augs > 0
        # Start with least populated image
        # Take max_attempts to randomly fill each images with stolen birds
        # Loop over source images sorted from least to most populous
        # Loop back to first image if all images exhausted
        # Break once augmentation target is met
        aug_idents = []
        bird_counter = 0
        birds_rehomed = 0
        for key in image_dict:
            
            # Read datum
            base_datum = image_dict[key]

            # Skip over unlabelled entries
            if len(base_datum.class_ids) == 0: continue

            # Load an image
            base_img = cv2.imread(base_datum.img_path)
                
            # Create base augmentations - rotate 90, 180, 270 degrees, mirror all 4 rotations on both x and y
            # results in 12 total images from base image
            aug_imgs = [copy.copy(base_img)]
            aug_labels = [copy.copy(base_datum.boxes)]
            
            # Rotate images and append results
            rotated_images, rotated_labels = rotate_aug(aug_imgs[0], aug_labels[0], rotations)
            aug_imgs += rotated_images
            aug_labels += rotated_labels

            # Debug
            #for i in range(len(aug_imgs)): draw_boxes(aug_imgs[i], aug_labels[i])
            
            # Mirror images and append results
            for i in range(len(aug_imgs)):
                mirrored_images, mirrored_labels = mirror_aug(aug_imgs[i], aug_labels[i], mirror_axis)
                aug_imgs += mirrored_images
                aug_labels += mirrored_labels
            
            # Generate idents for new images ### TODO include main loop number to make new idents unique
            aug_idents = []
            for i in range((len(rotations)+1) * (len(mirror_axis)+1)):
                aug_idents.append(base_datum.ident + "_aug-" + str(bird_class) + "-" + str(i))
            
            # Debug
            #for i in range(len(aug_imgs)):
            #    for box in aug_labels[i]:
            #        cv2.rectangle(aug_imgs[i], (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            #    cv2.imshow("test", aug_imgs[i])
            #    if cv2.waitKey(0) == ord('q'):
            #        cv2.destroyAllWindows()
            #        sys.exit()

            # Iterate over augmented images, re-home birds in each new image
            for aug_idx in range(len(aug_imgs)):
                
                # Attempt to randomly place n clone birds inside augmented images
                collision_labels = copy.copy(aug_labels[aug_idx])
                for attempt in range(max_attempts):

                    # Get size of bird
                    bird_shape = bird_clones[bird_counter].shape
                    
                    # Randomly generate an upper left coordinate and box
                    ry = random.randint(bird_shape[0], base_datum.img_size[0]-bird_shape[0])
                    rx = random.randint(bird_shape[1], base_datum.img_size[1]-bird_shape[1])
                    attempt_box = [rx, ry, rx+bird_shape[1], ry+bird_shape[0]]

                    # Inflate attempt box to buffer collision detection
                    collision_box = inflate_box(attempt_box, bird_shape, base_datum.img_size, collision_tol)
                    
                    # Check for collision with exisitng boxes
                    if check_collision(collision_box, collision_labels) == False:
                        
                        # No collisions, place bird in image
                        put_bird(aug_imgs[aug_idx], bird_clones[bird_counter], attempt_box, blend_inner, blend_outer)

                        # Deflate attempt box to account for thief inflate
                        attempt_box = inflate_box(attempt_box, bird_shape, base_datum.img_size, -thief_inflate)
                        aug_labels[aug_idx].append(attempt_box)

                        # Add collision box to list so future placement can avoid this bird
                        collision_labels.append(collision_box)
                        
                        # Iterate bird counters, loop if necessary
                        bird_counter+=1
                        birds_rehomed+=1
                        if bird_counter >= n_stolen: bird_counter = 0
                        
                        # Debug
                        #cv2.imshow("test", aug_imgs[aug_idx])
                        #cv2.imshow("testas", bird_clones[bird_counter])
                        #if cv2.waitKey(0) == ord('q'):
                        #    cv2.destroyAllWindows()
                        #    sys.exit()

                    # That spots taken! Try again!
                    else:
                        continue

                # Check if augmentation target has been met for this class
                if birds_rehomed >= class_stats[bird_class][1]:
                    
                    # Yay! delete superflous augs and proceed to exit this iteration
                    del aug_imgs[aug_idx:]
                    del aug_labels[aug_idx:]
                    del aug_idents[aug_idx:]
                    break

            # Generate label files for new images, save image, save statistics
            for aug_idx in range(len(aug_imgs)):
                
                # Convert all boxes to yolo format
                for i, box in enumerate(aug_labels[aug_idx]): aug_labels[aug_idx][i] = yolo_box(base_datum.img_size, box)

                # Write output file
                label_out_path = os.path.join(src_cfg['Labels'], aug_idents[aug_idx] + ".txt")
                with open(label_out_path, "w") as f:
                    
                    # Write boxes pre-exisiting in image
                    n_pre = len(base_datum.class_ids)
                    for i in range(n_pre):
                        f.write("{} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(base_datum.class_ids[i],
                            aug_labels[aug_idx][i][0],
                            aug_labels[aug_idx][i][1],
                            aug_labels[aug_idx][i][2],
                            aug_labels[aug_idx][i][3]
                        ))

                    # Write new boxes
                    for i in range(n_pre, len(aug_labels[aug_idx])):

                        f.write("{} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(bird_class,
                            aug_labels[aug_idx][i][0],
                            aug_labels[aug_idx][i][1],
                            aug_labels[aug_idx][i][2],
                            aug_labels[aug_idx][i][3]
                        ))

                # Close the file
                f.close()

                # Save the image
                image_out_path = os.path.join(src_cfg['Images'], aug_idents[aug_idx] + ".png")
                cv2.imwrite(image_out_path, aug_imgs[aug_idx])

                # Debug display final image with all bounding boxes
                #draw_boxes(aug_imgs[aug_idx], aug_labels[aug_idx])

            # Stats display
            print("Class '{}' | {} / {}...          \r".format(bird_class, birds_rehomed, class_stats[bird_class][1]), end="")
             
            # Check if augmentation target has been met, clear counter, break image loop
            if birds_rehomed >= class_stats[bird_class][1]:
                sys.stdout.write("\033[K")
                print("Target for class '{}' achieved!".format(bird_class))
                break

        # Check if augmentation target has been met, start next main class loop
        if birds_rehomed >= class_stats[bird_class][1]:
            continue
        else:
            # Restart main loop ### TODO
            print("WARNING: Failed to reach augmentation target in single pass! Go ask Matt to implement multi-pass!")
            print("Reached {} / {} for class '{}'!!!\n".format(birds_rehomed, class_stats[bird_class][1], bird_class))
            break

class LabelledImage():
    def __init__(self, ident, img_path, img_size, class_ids, boxes, n_inst):
        # Save vars
        self.ident = ident
        self.img_path = img_path
        self.img_size = img_size
        self.class_ids = class_ids
        self.boxes = boxes
        self.n_inst = n_inst

def draw_boxes(src_img, boxes):
    """
    Debug function to show bounding boxes on an image
    param src_img: image to draw boxes on
    param boxes: boxes to draw on image in cv or yolo format
    """
    
    # Copy inputs to avoid contamination
    test_img = copy.copy(src_img)
    test_boxes = copy.copy(boxes)

    # Iterate over boxes
    for i, box in enumerate(test_boxes):

        # Convert to cv box if in yolo format
        if sum(box) <= 4: box = cv_box(test_img.shape[:2], box)

        # Draw rectangle
        test_img = cv2.rectangle(test_img, (box[0], box[1]), (box[2], box[3]), (190, 40, 150), 2)
    
    # Display result
    cv2.imshow("Boxes", test_img)
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()
        sys.exit()

def put_bird(src_img, bird, box, blend_inner, blend_outer):
    """
    Place a cropped bird into a destination image and apply a blur arround the edge of the paste
    Assumes ul is safe
    Operation is in place on source image
    param src_img:          image to place the bird into
    param bird:             image of bird to place (must be smaller than src_img)
    param box:              target bounding cv box of bird [x_min, y_min, x_max, y_max]
    param blend_inner:      percent size to inflate into box
    param blend_outer:      percent size to inflate out of box
    """
    
    # Copy bird into image
    src_img[box[1]:box[3], box[0]:box[2]] = bird
    
    # Create blur mask
    outer = inflate_box(box, bird.shape, src_img.shape, blend_outer)
    inner = inflate_box(box, bird.shape, src_img.shape, -blend_inner)
    blur_width = [inner[1] - outer[1], outer[3] - inner[3], inner[0] - outer[0], outer[2] - inner[2]]
    
    # Take segment containing placed bird from image to blur
    blur_sel = copy.copy(src_img[outer[1]:outer[3], outer[0]:outer[2]])
    blur = cv2.GaussianBlur(blur_sel, (21,21), 11)
    
    # Create blur mask with outer rim to blur to outer image with small kernel blur
    rim_mask = np.zeros([outer[3]-outer[1], outer[2]-outer[0], 3], dtype=np.uint8)
    rim = int(min(rim_mask.shape) * 0.02)
    rim_mask = cv2.rectangle(rim_mask, (0,0), (rim_mask.shape[1]-1, rim_mask.shape[0]-1), (255, 255, 255), rim)
    rim_mask = cv2.GaussianBlur(rim_mask, (9,9), 5)
    
    # Create center mask and blur with large kernel
    cnt_mask = np.zeros([outer[3]-outer[1], outer[2]-outer[0], 3], dtype=np.uint8)
    cnt_mask[blur_width[0]:-blur_width[1], blur_width[2]:-blur_width[3], :] = 255
    cnt_mask = cv2.GaussianBlur(cnt_mask, (21,21), 11)

    # Combine masks
    mask = np.where(cnt_mask == 0, rim_mask, cnt_mask)
    
    # Create alpha mask and blend original selection and blurred version according to alpha gradient
    alpha = mask / 255.0
    blend = cv2.convertScaleAbs(blur*(1-alpha) + blur_sel*alpha)

    # Place Segment in image
    src_img[outer[1]:outer[3], outer[0]:outer[2]] = blend

    # Debug
    #cv2.imshow("bird", bird)
    #cv2.imshow("mask", mask)
    #cv2.imshow("blend", blend)
    #cv2.imshow("blur", blur)
    #cv2.imshow("blur_sel", blur_sel)
    #cv2.imshow("src", src_img)
    #if cv2.waitKey(0) == ord('q'):
    #    cv2.destroyAllWindows()
    #    sys.exit()

def inflate_box(src_box, box_shape, img_shape, pct):
    """
    Inlfate a bounding box within image_size bounds by pct percentage
    param src_box: bounding box to be inflated
    param box_shape: shape of box [m, n]
    param img_shape: shape of parent image [m, n]
    param pct: percentage to inflate box by
    """
    
    # Copy to avoid overwrite of src
    box = copy.copy(src_box)

    #Inflate box
    box[0] = round(box[0] - (box_shape[1] * pct / 2))
    box[1] = round(box[1] - (box_shape[0] * pct / 2))
    box[2] = round(box[2] + (box_shape[1] * pct / 2))
    box[3] = round(box[3] + (box_shape[0] * pct / 2))

    # Constrain box
    box[0] = max(box[0], 0)
    box[1] = max(box[1], 0)
    box[2] = min(box[2], img_shape[1])
    box[3] = min(box[3], img_shape[0])
    
    # Return box
    return box

def check_collision(candidate_box, boxes):
    """
    Check for collision between candidate box and multiple exisitng boxes
    param candidate_box: a cv format [x_min, y_min, x_max, y_max] box to check against existing boxes
    param boxes: a list of cv format [x_min, y_min, x_max, y_max] boxes
    return: True if collision detected, otherwise False
    """
    
    # Check for collision
    for i, box in enumerate(boxes):
        
        # Calculate metrics
        xaxbn = candidate_box[2] - box[0]
        xanbx = candidate_box[0] - box[2]
        yaxbn = candidate_box[3] - box[1]
        yanbx = candidate_box[1] - box[3]
        
        # A single negative value means overlap in ranges
        x = False
        y = False
        if (xaxbn < 0) != (xanbx < 0): x = True
        if (yaxbn < 0) != (yanbx < 0): y = True
        
        # If both x and y ranges overlap thats a collision!
        if x and y:
            return True

    # If all test pass return False
    return False

def mirror_aug(src_image, boxes, axis):
    """
    Create mirrored coppies of a source image and corresponding label files
    param src_image: image to operate on
    param boxes: list of boxes present in cv format [[x_min, y_min, x_max, y_max], ...]
    param axis: list of axis to flip image on
    return: [images], [labels]
    """
    
    images = []
    labels = []
    for ax in axis:
        
        # Create mirrored image
        img_mirrored = np.ascontiguousarray(np.flip(src_image, axis=ax))
        img_shape = img_mirrored.shape

        # Mirror bounding boxes
        labels_mirrored = []
        for box in boxes:
            
            # Copy box to avoid contaminating source
            inv_box = copy.copy(box)

            # Invert box
            if ax == 0:
                inv_box[1] = img_shape[0] - box[1]
                inv_box[3] = img_shape[0] - box[3]
            elif ax == 1:
                inv_box[0] = img_shape[1] - box[0]
                inv_box[2] = img_shape[1] - box[2]
           
            # Sort box
            x_min = min(inv_box[0], inv_box[2])
            y_min = min(inv_box[1], inv_box[3])
            x_max = max(inv_box[0], inv_box[2])
            y_max = max(inv_box[1], inv_box[3])

            # Reform ordered box
            inv_box = [x_min, y_min, x_max, y_max]
            
            # Add box in cv form to list
            labels_mirrored.append(inv_box)
            
            # Draw inv_box
            #cv2.rectangle(img_mirrored, (inv_box[0], inv_box[1]), (inv_box[2], inv_box[3]), (255, 0, 0), 2)
            #cv2.imshow("test_mirror", img_mirrored)
            #cv2.imshow("src", src_image)
            #if cv2.waitKey(0) == ord('q'):
            #    cv2.destroyAllWindows()
            #    sys.exit()

        # Save results
        images.append(img_mirrored)
        labels.append(labels_mirrored)
    
    # Return mirrored images and labels
    return images, labels

def rotate_aug(src_img, boxes, rotations):
    """
    Create rotated coppies of a source image and corresponding label files
    param src_img: image to operate on
    param boxes: list of boxes present in cv format [[x_min, y_min, x_max, y_max], ...]
    param rotations: list of rotations to implement (in degrees)
    return: [images], [labels]
    """
    
    images = []
    labels = []
    for angle in rotations:
        
        # Create rotated image
        img_rotated = copy.copy(src_img)
        img_rotated = np.ascontiguousarray(np.rot90(img_rotated, int(angle / 90), (0,1)))
        
        # Debug
        #cv2.imshow("test_rot", img_rotated)
        #if cv2.waitKey(0) == ord('q'):
        #    cv2.destroyAllWindows()
        #    sys.exit()

        # Rotate bounding boxes
        labels_rotated = []
        
        # Rotation Matrix
        theta = np.deg2rad(360-angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        o = np.asarray(src_img.shape[:2]) / 2
        for box in boxes:
            
            # Assert square image
            assert src_img.shape[0] == src_img.shape[1], "Image must be square"

            # Apply rotation matrix and offset origin to each point
            points = np.asarray((box[:2], box[2:]))
            points_rot = np.squeeze((R @ (points.T-o.T) + o.T).T)

            # Sort points to be [[x_min, y_min], [x_max, y_max]]
            points_rot = np.sort(points_rot, axis=0).astype(np.uint32)

            # Add box in cv form to list
            rot_box = [points_rot[0][0], points_rot[0][1], points_rot[1][0], points_rot[1][1]]
            labels_rotated.append(rot_box)
            
            # Debug
            #cv2.imshow("src_img", src_img)
            #draw_boxes(img_rotated, [rot_box])
             
        # Save results
        images.append(img_rotated)
        labels.append(labels_rotated)

    # Return rotated images and labels
    return images, labels 

def hue_circular_mean(hue):
    
    # Convert to radians
    radians = np.radians(hue, dtype=np.float64)
    
    # Calculate the sum of sin and cos values
    sin_sum = np.sum(np.sin(radians))
    cos_sum = np.sum(np.cos(radians))
    
    # Calculate the circular mean using arctan2
    mean_rad = np.arctan2(sin_sum, cos_sum)
    
    # Convert the mean back to 0-180
    mean_hue = (np.degrees(mean_rad) / 2) % 180

    return mean_hue

def validate_src_dir(src_path):
    """ 
    Check that a given source directory is of the format:
    \src_path
        \images_train       - containing lablled images (png)
        \labels_train       - containing yolo format labels (txt)
    """
    
    # Define keys
    keys = ['Source Root', 'Images', 'Labels']

    # Define dirs
    dirs = [src_path,
            os.path.join(src_path, "images_train"),
            os.path.join(src_path, "labels_train")]

    # Create dict
    in_dict = dict(zip(keys, dirs))

    # Check dirs exist
    for folder in in_dict.values():
        if not os.path.exists(folder):
            print("ERROR, source '{}' invalid, '{}' not found! EXITING...\n".format(src_path, folder))
            sys.exit()

    # Retrun validated paths
    return in_dict

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
    
    # Convert to yolo format and constrain
    width = x_max - x_min
    height = y_max - y_min
    x_center = x_min + (width / 2)
    y_center = y_min + (height / 2)

    # Return in yolo format
    return [x_center, y_center, width, height]

def cv_box(image_shape, box):
    """
    Helper function to create opencv boxes
    param image_shape: [rows, columns, ...], image shape
    param box: bounding box in normalised image form (0-1) [cx, cy, width, height]
    return: bounding box in pixel coordinates [x_min, y_min, x_max, y_max]
    """
    
    # Convert to opencv format
    x_min = box[0] - (box[2] / 2)
    x_max = box[0] + (box[2] / 2)
    y_min = box[1] - (box[3] / 2)
    y_max = box[1] + (box[3] / 2)
    
    # Denormalise and constrain
    x_min = max(int(x_min * image_shape[1]), 0)
    y_min = max(int(y_min * image_shape[0]), 0)
    x_max = min(int(x_max * image_shape[1]), image_shape[1])
    y_max = min(int(y_max * image_shape[0]), image_shape[0])
    

    # Return in yolo format
    return [x_min, y_min, x_max, y_max]

if __name__ == '__main__':
    main()
