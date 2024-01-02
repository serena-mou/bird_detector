import glob
import cv2
import os
im_folder = '/home/serena/Data/Turtles/turtle_datasets/031216amnorth/split_data_1_class/test/images/*.jpg'
label_folder ='/home/serena/Data/Turtles/turtle_datasets/031216amnorth/split_data_1_class/test/labels/*.txt'

im_size = (1920,1080)


all_im = sorted(glob.glob(im_folder))
all_label = sorted(glob.glob(label_folder))
for i, im_file in enumerate(all_im):
    label_file = all_label[i]
    print(im_file)
    print(label_file)

    im = cv2.imread(im_file)
    with open(label_file, 'r') as f:
        boxes = f.readlines()
    m = 0
    for i, b in enumerate(boxes):
        data = b.split(' ')
        x = float(data[1])
        y = float(data[2])
        w = float(data[3])
        h = float(data[4])
        x1 = int(x*im_size[0])
        y1 = int(y*im_size[1])
        x2 = int(x1 + w*im_size[0])
        y2 = int(y1 + h*im_size[1])
        m=i    
        cv2.rectangle(im,(x1,y1),(x2,y2),(0,0,255),2)
    print(m)
    cv2.imshow('all rects', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()