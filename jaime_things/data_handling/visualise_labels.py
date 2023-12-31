import glob
import cv2
import os
im_folder = '/home/serena/Data/Birds/common_noddy/CN_yolov8_50/split_data/train/images/*.jpg'
label_folder = '/home/serena/Data/Birds/common_noddy/CN_yolov8_50/split_data/train/labels/*.txt'
out_folder = '/home/serena/Data/Birds/common_noddy/CN_yolov8_50/train_raw_data'

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
        x1 = int(x*416)
        y1 = int(y*416)
        x2 = int(x1 + 70)
        y2 = int(y1 + 70)
        m=i    
        cv2.rectangle(im,(x1,y1),(x2,y2),(0,0,255),2)
    name = im_file.split('/')[-1]
    cv2.imwrite(os.path.join(out_folder,name), im)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
