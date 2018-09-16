# 该文件用来调整图片的大小

# -*- coding: utf-8 -*-
from glob import glob
import cv2

imagelist = glob('images/*.jpg')
print(len(imagelist))
for image_name in imagelist:
    print(image_name)
    image = cv2.imread(image_name)
    #cv2.imshow("image", image)
    #cv2.waitKey(0)
    print(image.shape)
    h,w,_ = image.shape
    #print("height", h)
    #print("weight", w)
    if h > w:
        f = h / 1024
        new_w = int(w // f)
        img = cv2.resize(image, (new_w, 1024), interpolation=cv2.INTER_CUBIC)
        print("new_shape", img.shape)
    else:
        f = w / 1024
        new_h = int(h // f)
        img = cv2.resize(image, (1024, new_h), interpolation=cv2.INTER_CUBIC)
        print("new_shape", img.shape)
        #cv2.imshow("new_image", img)
        #cv2.waitKey(0)
    image_name = image_name.split("\\")[-1]
    cv2.imwrite("resized/resize_" + image_name, img)
    #res = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_CUBIC)
