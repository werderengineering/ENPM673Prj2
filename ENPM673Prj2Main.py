import numpy as np
import cv2
from __main__ import *
import imutils
import math

from imageFileNames import imagefiles


print('Imports Complete')

print('CV2 version')
print(cv2.__version__)

prgRun = True

def main(prgRun):
    problem = 2

    #Correct image
    if problem ==1:

        video = cv2.VideoCapture('Night Drive - 2689.mp4')

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
            # I did this
            ###################Hough##########################################

            ###################Homography and Impose##########################

            cv2.imshow('DWF', clnframe)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break


    #Lane finder Image set
    elif problem ==2:
        directory='./data'
        # directory=str(input('What is the name of the folder with the images? Note, this should be entered as"/folder": \n'))

        print(directory)

        imageList=imagefiles(directory)

        for i in range(len(imageList)):

            frameDir=directory+'/'+imageList[i]
            frame=cv2.imread(frameDir)

            ##########################Correct frame###########################

            ############################Histogram Equalization################

            ####################Contour#######################################

            #####################Homography and dewarp########################

            ###################Hough##########################################

            ###################Homography and Impose##########################

            cv2.imshow('image', frame)
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