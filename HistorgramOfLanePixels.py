import numpy as np
import cv2
from __main__ import *
import imutils


def historgramOnYAxis(img_gray):
    """calculate the historgram of pixel value along y axis"""
    # hist = np.zeros(img_gray.shape[1])
    hist = np.mean(img_gray, axis=0)
    assert hist.shape[0] == img_gray.shape[1]
    return hist


def debug():
    img_path = "./data/0000000000.png"
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    historgramOnYAxis(img_gray)


debug()
