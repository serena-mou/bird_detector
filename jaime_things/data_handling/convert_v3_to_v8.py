import os
import re
import numpy as np
# Load yolov3 annotations txt file
base = '/home/serena/Data/Birds/lf_mb_rfb/50x50'  
v3_annotations = base+'/raw_data_all/_annotations.txt'
output_folder = base+'/obj_train_data/'
train_txt = base+'/train.txt'
im_size = (412,412) # x,y
box_size = 50


if not os.path.exists(output_folder):
    os.makedirs(output_folder)


with open(v3_annotations, 'r') as f:
    lines = f.readlines()

#print(lines[0])
#train_lines = []
for line in lines:
    #img_name = str(line)[0].split('/')[-1]
    
    line_str = re.split(' ',line)
    # print(line_str)
    img_name = line_str[0].split('/')[-1]
    bbs = line_str[1:]
    #print(bbs)
    file_name = img_name.split('.')[0]
    output_file = os.path.join(output_folder,file_name+'.txt')
    write_lines = []
    for box in bbs:
        # convert to class x y w h normalised
        #box = np.array(box)
        box = box.split(',')
        x1 = int(box[0])+(box_size/2)
        y1 = int(box[1])+(box_size/2)
        cls = str(box[4][0])

        x = str(x1/im_size[0])
        y = str(y1/im_size[1])
        w = str(box_size/im_size[0])
        h = str(box_size/im_size[1])

        write_lines.append(cls + ' '+x +' '+ y +' '+ w +' '+ h)
        
    #print(output_file)
    #print(write_lines)
    
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
