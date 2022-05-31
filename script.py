import os
import cv2

path="./GaitDataset/001/nm-03/090"

img_list=[]

for file in os.listdir(path):
    
    img=cv2.imread(path+"/"+file,cv2.IMREAD_GRAYSCALE)
    ret,img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)

    x,y,w,h = cv2.boundingRect(img)

    img = cv2.resize(img[y:y+h,x:x+w],(70,210))

    cv2.imshow("Image",img)
    cv2.waitKey()

