
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Dropout

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import cv2

cap = cv2.VideoCapture(0)
count = 1

while count!=1000:
    ret, frame = cap.read()
    cv2.rectangle(frame, (300,300), (100,100), (0,255,0),0)
    crop_img = frame[100:300, 100:300]
    
    img = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
    
    block_size = 513
    constant = 2
    
    th1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant)
    
    
    test = cv2.resize(th1,(64,64))
    test = cv2.cvtColor(test,cv2.COLOR_GRAY2BGR,dstCn = 3)

    cv2.imwrite("hand%d.jpg" % count , th1)
    
    cv2.imshow('original',frame)
    cv2.imshow('saving',th1)
    cv2.imshow('data feed',test)
    count += 1
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    

cv2.destroyAllWindows()
cap.release()