# save output from a trained model

from PIL import Image
from ultralytics import YOLO
import glob
import os
import numpy as np

# Load a pretrained YOLOv8n model
model = YOLO('../ultralytics/runs/detect/train35/weights/best.pt')

base = '/home/serena/Data/sctld/sctld-round2/split_data/'

SAVE = True
# Run inference on 'bus.jpg'
suffixes = ['*.JPG', '*.jpeg', '*.jpg', '*.png']
image_path = base + 'test/images/'
#image_path = base + 'test/images/*.JPG'
#image_path = base + 'split_data/test/images/*.jpg'
out_folder = base + 'test_pred35/'
image_files = []

if not os.path.isdir(out_folder):
    os.makedirs(out_folder)

for suffix in suffixes:

    image_files = image_files + glob.glob(image_path+'/'+suffix)

print(len(image_files))

for im in image_files:
    results = model(im, conf=0.1)  # results list
    im_out = im.split('/')[-1] 
    # Show the results

    txt_out = im_out[0:-4]+'.txt'
    print(txt_out)
    classes = results[0].boxes.cls.cpu().numpy()
    boxes = results[0].boxes.xywhn.cpu().numpy()
    

    
    im_array = results[0].plot(line_width=10)  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    
    if len(classes) > 0:
        with open(os.path.join(out_folder,txt_out),'w') as f:
            for i,c in enumerate(classes):
                cx,cy,w,h = list(boxes[i])
                line = "%d %0.4f %0.4f %0.4f %0.4f"%(c,cx,cy,w,h)
                #print(line)
                f.write("%s\n"%line)
    #im.show()  # show image
    #input()
    if SAVE:
        im.save(os.path.join(out_folder, im_out))  # save image
