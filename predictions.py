import cv2
import os
import numpy


cap = cv2.VideoCapture(0)
ret, frame = cap.read()

labels = {0:'A', 1:'B', 2:'C'}

while True:    
    x1 = 10
    y1 = 10
    x2 = int(0.5 * frame.shape[1])
    y2 = int(0.5 * frame.shape[0])
