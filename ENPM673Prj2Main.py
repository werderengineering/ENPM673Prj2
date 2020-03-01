import numpy as np
import cv2
import imutils
import math

from imageFileNames import imagefiles
from imagecorrection import *
import HomoCalculation
from HistorgramOfLanePixels import *
from fitlines import *
import random

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

            ###################Output Imagery##########################
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
        homo = HomoCalculation.homo_problem2()
        """process each image individually"""
        for i in range(len(imageList)):
            frameDir = directory + '/' + imageList[i]
            frame=cv2.imread(frameDir)
            # frame = imutils.resize(frame, width=320, height=180)

            ##########################Correct frame###########################

            ##########################Thresh frame###########################
            grayframe=grayscale(frame)
            binaryframe = yellowAndWhite(frame)
            ############################Region ################
            region = process_image(binaryframe)
            #####################Homography and dewarp########################

            """the next line give you a flat view of current frame"""
            flatfieldBinary = cv2.warpPerspective(region, homo, (frame.shape[0], frame.shape[1]))
            flatBGR= cv2.warpPerspective(frame, homo, (frame.shape[0], frame.shape[1]))


            ####################Contour#######################################
            cnts, hierarchy = cv2.findContours(flatfieldBinary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            bincntframe = cv2.drawContours(flatfieldBinary, cnts,-5, (255), 5)
            cv2.imshow('flatfield binary', bincntframe)

            # cntframe = cv2.drawContours(flatBGR, cnts,-5, (255, 0, 0), 5)

            ###################Draw Lines##########################################

            flatBGRLanes,LeftLines,RightLines=MarkLanes(bincntframe, flatBGR, frame)

            ###################Homography and Impose##########################

            # homo=np.linalg.inv(homo)
            # FinalBGR= cv2.warpPerspective(flatBGR, homo, (frame.shape[0], frame.shape[1]))


            ###################Output Imagery##########################
            cv2.imshow('Working Frame', flatBGRLanes)
            cv2.imshow('Final Frame', frame)


            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            if flag:
                cv2.imshow('unwarped video', img_unwarped)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break



    #Lane Finder Challenge vid
    elif problem == 3:

        video = cv2.VideoCapture('challenge_video.mp4')
        homo = HomoCalculation.homo_problem3()
        # Read until video is completed
        while (video.isOpened()):
            # Capture frame-by-frame
            ret, frame = video.read()
            if ret == True:
                # frame = imutils.resize(frame, width=320, height=180)
                # frame = frame[int(frame.shape[0] / 2) + 70:, :]
                ogframe = frame
                clnframe = frame
                resetframe = frame

                ##########################Correct frame###########################
                binaryframe = yellowAndWhite(frame)

                ############################Region ################
                region = process_image(binaryframe)
                #####################Homography and dewarp########################

                """the next line give you a flat view of current frame"""
                flatfieldBinary = cv2.warpPerspective(region, homo, (frame.shape[0], frame.shape[1]))
                flatBGR = cv2.warpPerspective(frame, homo, (frame.shape[0], frame.shape[1]))

                ####################Contour#######################################
                cnts, hierarchy = cv2.findContours(flatfieldBinary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # cv2.imshow('flatfield', flatfieldBinary)
                cntframe = cv2.drawContours(flatBGR, cnts, -5, (255, 0, 0), 5)

                ###################Draw Lines##########################################
                # hist=historgramOnYAxis(grayframe)
                # testpoints=np.array


                ###################Homography and Impose##########################



                ###################Output Imagery##########################

                frame = imutils.resize(frame, width=320, height=180)
                cntframe = imutils.resize(cntframe, width=320, height=180)
                flatfieldBinary = imutils.resize(flatfieldBinary, width=320, height=180)


                # cv2.imshow('Original frame', frame)
                cv2.imshow('Working frame', cntframe)
                cv2.imshow('Flat frame', flatfieldBinary)

                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
    cv2.destroyAllWindows()
    prgRun=False
    return prgRun







print('Function Initializations complete')

if __name__ == '__main__':
    print('Start')
    prgRun = True
    while prgRun == True:
        prgRun = main(prgRun)

    print('Goodbye!')
    cv2.destroyAllWindows()