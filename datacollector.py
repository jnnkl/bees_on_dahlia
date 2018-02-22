#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 12:22:36 2017

@author: bombus
"""
#warning this script works for python 2.7 but not for python 3.6
#import numpy as np
import cv2
import time
import datetime
import os

#make a folder with the fname first.
fname='test'
 
if not os.path.exists(fname):
    os.makedirs(fname)

#open the cap and get the time
start=time.time()
cap = cv2.VideoCapture(0)

#save an image every second for 10000 seconds
while(cap.isOpened() and time.time()-start<10000):
    for i in range(20):
        print(i,cap.get(i))
    cap.set(10,1.0)
    ret, frame = cap.read()
    timestr = datetime.datetime.now().strftime("%Y%m%d_%H%M_%S")
    if ret==True:
       


    # write the  frame
        cv2.imwrite(fname+"/"+fname+timestr+'.jpg',frame)
        

        cv2.imshow('frame',frame)
        time.sleep(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
#out.release()
cv2.destroyAllWindows()
