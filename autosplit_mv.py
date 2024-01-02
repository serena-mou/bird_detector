import shutil
import os

os.chdir('/home/serena/Data/Birds/Serena/MBay/Trip_1_cut_512_box50_border10_rgb/')

files_in = [ './autosplit_train.txt', './autosplit_val.txt', './autosplit_test.txt' ]

for i, FILE_IN in enumerate(files_in):

    if i == 0:
        FOLDER_OUT_im = './split_data/train/images'
        FOLDER_OUT_txt = './split_data/train/labels'

    if i == 1:
        FOLDER_OUT_im = './split_data/valid/images'
        FOLDER_OUT_txt = './split_data/valid/labels'

    if i == 2:
        FOLDER_OUT_im = './split_data/test/images'
        FOLDER_OUT_txt = './split_data/test/labels'

    if not os.path.exists(FOLDER_OUT_im):
        os.makedirs(FOLDER_OUT_im)

    if not os.path.exists(FOLDER_OUT_txt):
        os.makedirs(FOLDER_OUT_txt)


    with open(FILE_IN, 'r') as file:
        data = file.read().splitlines()

        print(len(data))
        print(data[0].split('/'))
        

        for d in data:
            cls = d.split('/')[2]
            fn_im = d.split('/')[-1]
            IN_im = d
            
            
            IN_txt = d.split('.')[1]
            #print(IN_txt)
            #IN_txt = IN_txt[4:]
            IN_txt = '.'+IN_txt+'.txt' 
            
            in_txt = fn_im.split('.')
            fn_txt = in_txt[0]+'.txt'
            
            OUT_im = os.path.join(FOLDER_OUT_im,fn_im)
            
            OUT_txt = os.path.join(FOLDER_OUT_txt,fn_txt)
            #print(IN_im, OUT_im)
            #print(IN_txt, OUT_txt)
            shutil.move(IN_im,OUT_im)
            if os.path.isfile(IN_txt):
                shutil.move(IN_txt,OUT_txt)

