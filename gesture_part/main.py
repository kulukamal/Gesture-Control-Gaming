# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 21:21:04 2018

@author: kamal
"""
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Dropout

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import cv2
from keras.models import load_model

from pyautogui import press
import time
cap = cv2.VideoCapture(0)
count = 1
classifier = load_model('model2_99.h5')
while True:
    cv2.waitKey(10)
    ret, frame = cap.read()
    cv2.rectangle(frame, (300,300), (100,100), (0,255,0),0)
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    block_size = 513
    constant = 2
    
    crop_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant)
    crop_img2 = cv2.cvtColor(crop_img,cv2.COLOR_GRAY2BGR)
    vision = crop_img[100:300, 100:300]
    
    test = cv2.resize(vision,(64,64))
    test = cv2.cvtColor(test,cv2.COLOR_GRAY2BGR,dstCn = 3)
    
    
    prediction = classifier.predict([[test]])
    if prediction == 1:
        txt = 'JUMP'
        press('space')
    elif prediction == 0:
        txt = ' '
    else:
        txt = ' '
    cv2.putText(frame,text = txt,org = (60,375),fontScale = 3,fontFace = 2 , color = (255,255,255))
  
    final_img = np.concatenate((frame,crop_img2),axis=1)
    cv2.imshow('original',final_img)
    cv2.imshow('gesture',vision)
    count += 1
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
    

cv2.destroyAllWindows()
cap.release()