import os
import re
import numpy as np
# Load yolov3 annotations txt file
v3_annotations = '/home/serena/Data/Birds/lesser_frigates/_annotations_50.txt'
output_folder = '/home/serena/Data/Birds/lesser_frigates/LF_yolov8_50/obj_train_data/'
# train_txt = '/home/serena/Data/Birds/common_noddy/CN_yolov8/train.txt'
im_size = (412,412) # x,y
box_size = 50

with open(v3_annotations, 'r') as f:
    lines = f.readlines()

#print(lines[0])
#train_lines = []
for line in lines:
    #img_name = str(line)[0].split('/')[-1]
    
    line_str = re.split('/| ',line)
    img_name = line_str[9]
    #print(img_name)
    bbs = line_str[10:]
    #print(img_name)
    file_name = img_name.split('.')[0]
    output_file = os.path.join(output_folder,file_name+'.txt')
    #print(output_file)
    write_lines = []
    for box in bbs:
        # convert to class x y w h normalised
        # print(box)
        #box = np.array(box)
        box = box.split(',')
        x1 = int(box[0])
        y1 = int(box[1])
        cls = '0'

        x = str(x1/im_size[0])
        y = str(y1/im_size[1])
        w = str(box_size/im_size[0])
        h = str(box_size/im_size[1])

        write_lines.append(cls + ' '+x +' '+ y +' '+ w +' '+ h)
        
        #print(train_txt)
    # print(write_lines)
    #print(output_file)
    with open(output_file, 'w') as o:
        for write_line in write_lines:
            o.write(write_line)
            o.write('\n')
    # train_lines.append('data/obj_train_data/'+img_name)
'''
with open(train_txt, 'w') as t:
    for train_line in train_lines:
        t.write(train_line)
        t.write('\n')
'''    
        #print(write_line)
    
    #bboxes = line[1:]
    #print(bbs)