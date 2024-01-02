import glob
import os
import numpy as np
import cv2



base = '/home/serena/Data/Birds/Serena/MBay/Trip_1_cut_512_box50_border10_rgb/split_data/'
birds = ['Brown Noddy (B)', 'Silver Gull (C)', 'Brown Booby (A)',
        'Onchoprion Sp. (D)', 'CT COLONY (FCOL)', 'CT OTHER (F)', 'Sterna Sp. (G)',
        'UNKNOWN (U)']
colours = [(0,110,170),(128,128,128),(75,25,230),(128,0,0),(25,230,230),(180,30,145),(60,180,60),(212,190,250)]
'''
legend = 255*np.ones(shape=[250,300,3],dtype=np.uint8)

for i in range(len(birds)): 

    row = (i*25)+25   
    col_circle = 5  
    col_text = 10
    cv2.circle(legend,(col_circle,row),5,colours[i],-1)
    cv2.putText(legend,birds[i],(col_text, row+10),cv2.FONT_HERSHEY_SIMPLEX,0.8,colours[i],2,cv2.LINE_AA)

cv2.imshow("named",legend)
cv2.waitKey(0)
input()
cv2.destroyAllWindows()

cv2.imwrite(os.path.join(base,'legend.png'),legend)
'''
box_sz = 50
save_path = os.path.join(base,'test_pred')
if not os.path.isdir(save_path):
    os.mkdir(save_path)

SHOW = False
SAVE = True

ims = sorted(glob.glob(base+'test/images/'+'*.jpg'))

for im in ims:
    image = cv2.imread(im)
    shape = image.shape
    img_sz = shape[0]
    file_name = im.split('/')[-1]
    name = file_name.split('.')[0]+'.txt'

    txt_path = os.path.join(base,'test_pred',name)
    #print(txt_path)
    if os.path.isfile(txt_path):
        
        with open(txt_path,'r') as f:
            lines = f.readlines()
            for l in lines:

                cls,cx,cy,w,h = l.split(' ')
                cls = int(cls)
                cx = float(cx)
                cy = float(cy)
                top_left = (int(cx*img_sz-0.5*box_sz),int(cy*img_sz-0.5*box_sz))
                text_org = (int(cx*img_sz-0.5*box_sz),int(cy*img_sz-0.5*box_sz)-2)
                bottom_right = (int(cx*img_sz+0.5*box_sz),int(cy*img_sz+0.5*box_sz))
                mid = (int(cx*img_sz),int(cy*img_sz))
                color = colours[cls]
                # print(top_left, bottom_right, color)
                # cv2.rectangle(image,top_left, bottom_right, color, 2)
                cv2.circle(image,mid,3,color,-1)
                # cv2.putText(image,birds[cls],text_org,cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA)

    if SAVE:
        cv2.imwrite(os.path.join(save_path,file_name), image)

    if SHOW:
        cv2.imshow("named",image)
        cv2.waitKey(0)
        input()
        cv2.destroyAllWindows()

