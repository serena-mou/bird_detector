import cv2

in_im = '/home/serena/Data/Birds/lesser_frigates/LF_yolov8/split_data/test/images/LesserFrigatebird_12.jpg'
in_txt = '/home/serena/Data/Birds/lesser_frigates/LF_yolov8/split_data/test/labels/LesserFrigatebird_12.txt'

with open(in_txt,'r') as f:
    bbs = f.readlines()

image = cv2.imread(in_im)


for bb in bbs:
    data = bb.split(' ')
    x = float(data[1])
    y = float(data[2])
    x1 = int(x*416)
    y1 = int(y*416)
    x2 = int(x1 + 70)
    y2 = int(y1 + 70)

    cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


    