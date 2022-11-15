# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 14:24:30 2021

@author: Filipe Pacheco

Take a picture from Webcan via OpenCV2

"""

# Libraries - Imports
import matplotlib.pyplot as plt
import cv2
from skimage.transform import resize

# Create the video object
video = cv2.VideoCapture(0)

_, frame = video.read() # Saves the snapshot into a variable

plt.imshow(frame) # Original Snapshot
plt.imshow(resize(frame,(128,128,1))) # Edited Snapshot   