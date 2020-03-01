import numpy as np
import cv2
from __main__ import *
import imutils
import math

from imageFileNames import imagefiles
from imagecorrection import *
import HomoCalculation

from regionOfInterest import *

print('Imports Complete')

print('CV2 version')
print(cv2.__version__)

flag = False
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
            region=process_image(frame)
            ############################Histogram Equalization################

            #####################Homography and dewarp########################
            homo = HomoCalculation.homo_problem2()
            """the next line give you a flat view of current frame"""
            img_unwarped = cv2.warpPerspective(frame, homo, (frame.shape[0], frame.shape[1]))
            ####################Contour#######################################

            ###################Hough##########################################

            ###################Homography and Impose##########################

            cv2.imshow('Nicks Work', region)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

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
                # frame = imutils.resize(frame, width=320, height=180)
                frame = frame[int(frame.shape[0] / 2) + 70:, :]
                ogframe = frame
                clnframe = frame
                resetframe = frame

                ##########################Correct frame###########################
                binaryframe = yellowAndWhite(frame)
                # frame=binaryframe
                ############################Histogram Equalization################
                region = process_image(binaryframe)



                ####################Contour#######################################

                #####################Homography and dewarp########################
                # homo = HomoCalculation.homo()
                # """the next line give you a flat view of current frame"""
                # img_unwarped = cv2.warpPerspective(frame, homo, (frame.shape[0], frame.shape[1]))
                ###################Hough##########################################

                ###################Homography and Impose##########################

                cv2.imshow('Original frame', region)
                cv2.imshow('Working frame', binaryframe)
                # cv2.imshow('Flat frame', img_unwarped)

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