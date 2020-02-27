import numpy as np
import cv2
from __main__ import *
import imutils
import math


print('Imports Complete')

print('CV2 version')
print(cv2.__version__)

prgRun = True

def main(prgRun):
    video = cv2.VideoCapture('data_1.mp4')

    # Read until video is completed
    while (video.isOpened()):
        # Capture frame-by-frame
        ret, frame = video.read()
        if ret == True:
            frame = imutils.resize(frame, width=320)
            ogframe = frame
            clnframe = frame
            resetframe = frame

            ##########################Correct frame###########################




            ############################Histogram Equalization################


            ####################Contour#######################################

            #####################Homography and dewarp########################

            ###################Hough##########################################

            ###################Homography and Impose##########################




        cv2.imshow('DWF', clnframe)


    prgRun=False







print('Function Initializations complete')

if __name__ == '__main__':
    print('Start')
    while prgRun == True:
        prgRun = main(prgRun)

    print('Goodbye!')
    cv2.destroyAllWindows()