import glob
import os


os.chdir('/home/serena/Data/Birds/Serena/cut_partially_labelled/cvat_upload')

all_im = sorted(glob.glob('data/obj_train_data/*.jpg'))
print(all_im[0])

out_file = 'data/train.txt'

with open(out_file,'w') as f:
    for im in all_im:
        f.write(im+'\n')
