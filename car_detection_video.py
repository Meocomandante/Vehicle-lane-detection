from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import imageForms as iF
import math

parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--car_1_cascade', help='Path to cars cascade.',
                    default='C:\\Users\\ragma\\OneDrive\\Desktop\\TAPDI\\images\\video\\cars.xml')
parser.add_argument('--car_2_cascade', help='Path to cars cascade.',
                    default='C:\\Users\\ragma\\OneDrive\\Desktop\\TAPDI\\images\\video\\cars2.xml')
parser.add_argument('--car_3_cascade', help='Path to cars cascade.',
                    default='C:\\Users\\ragma\\OneDrive\\Desktop\\TAPDI\\images\\video\\cars3.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)

args = parser.parse_args()
car_1_cascade_name = args.car_1_cascade
car_2_cascade_name = args.car_2_cascade
car_3_cascade_name = args.car_3_cascade
car_1_cascade = cv.CascadeClassifier()  # importar face xml
car_2_cascade = cv.CascadeClassifier()  # importar eyes xml
car_3_cascade = cv.CascadeClassifier()  # importar eyes xml

# -- 1. Load the cascades
if not car_1_cascade.load(cv.samples.findFile(car_1_cascade_name)):
    print('--(!)Error loading car_1 cascade')
    exit(0)

if not car_2_cascade.load(cv.samples.findFile(car_2_cascade_name)):
    print('--(!)Error loading car_2 cascade')
    exit(0)

if not car_3_cascade.load(cv.samples.findFile(car_3_cascade_name)):
    print('--(!)Error loading car_3 cascade')
    exit(0)

def detectAndDisplay(frame, left_m, left_b, right_m, right_b):

    #frame_color= cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #frame_gray = cv.equalizeHist(frame_color)
    #get left line variables


    #y-m*x-b > 0 for left line

    # #-- Detect faces 1
    faces = car_1_cascade.detectMultiScale(frame,  1.2, 4 )
    for (x,y,w,h) in faces:
        if y < 300 and (y+h) > 60:
            pt1 = (x, y)
            pt2 = (x + w, y+h)
            #if center at bottom of image
            if (((y+ h) - left_m * (x+ w/2) - left_b > 0) and ((y+ h) - right_m * (x+ w/2) - right_b > 0) ):
                frame = cv.rectangle(frame, pt1, pt2, (0, 0 , 255), 3)
            else:
                frame = cv.rectangle(frame, pt1, pt2, (0, 255, 0), 3)

        # -- Detect faces 1
    eyes = car_2_cascade.detectMultiScale(frame,  1.3, 2 )
    for (x, y, w, h) in eyes:
        if y < 300 and (y+h) > 60:
            pt1 = (x, y)
            pt2 = (x + w, y + h)
            # if center at bottom of image
            if (((y+ h) - left_m * (x+ w/2) - left_b > 0) and ((y+ h) - right_m * (x+ w/2) - right_b > 0) ):
                frame = cv.rectangle(frame, pt1, pt2, (0, 0 , 255), 3)
            else:
                frame = cv.rectangle(frame, pt1, pt2, (0, 255, 0), 3)

        # -- Detect faces 1
    yes = car_3_cascade.detectMultiScale(frame, 1.3, 2)
    for (x, y, w, h) in yes:
        if y < 300 and (y+h) > 60:
            pt1 = (x, y)
            pt2 = (x + w, y + h)
            # if center at bottom of image
            if (((y+ h) - left_m * (x+ w/2) - left_b > 0) and ((y+ h) - right_m * (x+ w/2) - right_b > 0) ):
                frame = cv.rectangle(frame, pt1, pt2, (0, 0, 255), 3)
            else:
                frame = cv.rectangle(frame, pt1, pt2, (0, 255, 0), 3)

