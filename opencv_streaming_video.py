# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 10:07:08 2022

@author: Filipe Pacheco

Program to exhibit in real time the webcam video

"""

# Libraries - Import 
import cv2

# Create the video object
vidcap = cv2.VideoCapture(0)

if not vidcap.isOpened():
    raise IOError("Cannot open webcam")

while vidcap.isOpened(): # Keep the session opened until the 'ESC' button being pressed
    ret, frame = vidcap.read()
    frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', frame)
    c = cv2.waitKey(10)
    if c == 27:
        vidcap.release()
        cv2.destroyAllWindows()