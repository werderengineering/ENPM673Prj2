import numpy as np
import cv2
from __main__ import *
import imutils
import math

from imageFileNames import imagefiles
from imagecorrection import *
import HomoCalculation

print('Imports Complete')

print('CV2 version')
print(cv2.__version__)

flag = True
prgRun = True

def main(prgRun):
    problem=1

    #Correct image
    if problem ==1:

        video = cv2.VideoCapture('Night Drive - 2689.mp4')

        # Read until video is completed
        while (video.isOpened()):
            # Capture frame-by-frame
            ret, frame = video.read()
            if ret == True:
                frame = imutils.resize(frame, width=320, height=180)
                frame.shape
                ogframe = frame
                clnframe = frame
                resetframe = frame

            ##########################Correct frame###########################
            gamma=5
            sat=5
            hue=1
            contrast = 60

            frame = adjust_gamma(frame, gamma)
            frame = adjustSaturation(frame, hue, sat)
            frame=adjustContrast(frame, contrast)
            frame = cv2.bilateralFilter(frame, 15, 75, 75)






            cv2.imshow('DWF', frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break


    #Lane finder Image set
    elif problem == 2:
        directory = './data'
        # directory=str(input('What is the name of the folder with the images? Note, this should be entered as"/folder": \n'))

        print("Getting images from " + str(directory))
        imageList = imagefiles(directory)  # get a stack of images

        """process each image individually"""
        for i in range(len(imageList)):
            frameDir = directory + '/' + imageList[i]
            frame=cv2.imread(frameDir)
            # frame = imutils.resize(frame, width=320, height=180)

            ##########################Correct frame###########################

            ############################Histogram Equalization################

            #####################Homography and dewarp########################
            homo = HomoCalculation.homo()
            """the next line give you a flat view of current frame"""
            img_unwarped = cv2.warpPerspective(frame, homo, (frame.shape[0], frame.shape[1]))
            ####################Contour#######################################

            ###################Hough##########################################

            ###################Homography and Impose##########################

            if flag:
                cv2.imshow('unwarped video', img_unwarped)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break



    #Lane Finder Challenge vid
    elif problem == 3:

        video = cv2.VideoCapture('challenge_video.mp4')

        # Read until video is completed
        while (video.isOpened()):
            # Capture frame-by-frame
            ret, frame = video.read()
            if ret == True:
                frame = imutils.resize(frame, width=320, height=180)
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
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break


    prgRun=False







print('Function Initializations complete')
prgRun = True
if __name__ == '__main__':
    print('Start')
    while prgRun == True:
        prgRun = main(prgRun)

    print('Goodbye!')
    cv2.destroyAllWindows()