#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 19:41:48 2020

@author: rjn
"""
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# We point OpenCV's CascadeClassifier function to where our 
# classifier (XML file format) is stored
eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

try:
    while(True):
        _, frame = cap.read()  
        frame = cv2.flip(frame,1)
        
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                
        # Our classifier returns the ROI of the detected face as a tuple
        # It stores the top left coordinate and the bottom right coordiantes
        faces = face_classifier.detectMultiScale(gray, 1.15, 6) 
        
        # We iterate through our faces array and draw a rectangle
        # over each face in faces
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (127,0,255), 2)
            
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_classifier.detectMultiScale(roi_gray)
            
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(frame,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(0,0,255),2) 
                
        cv2.imshow('face detected',frame)
        
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
