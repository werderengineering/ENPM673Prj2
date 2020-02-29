import numpy as np
import perspective
import cv2
import imutils


def homoToResCenter(shape_res):
    if len(shape_res) == 3:
        shape_res = shape_res[0:2]
    shape_res = (np.asarray(shape_res) / 2).astype(int)
    src_corners_coordinates = eyeballLaneCorners()
    dest_corners_coordinates = perspective.square()
    dest_corners_coordinates, _ = perspective.shiftToASpot(dest_corners_coordinates, shape_res)
    homo, _ = cv2.findHomography(src_corners_coordinates, dest_corners_coordinates, 0)
    print(homo)
    return homo


def eyeballLaneCorners():
    img_path = "./data/0000000000.png"
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    centers_coordinates = [(512, 325), (760, 325), (886, 470), (274, 470)]
    for center_coordinates in centers_coordinates:
        img_gray = cv2.circle(img_gray, center_coordinates, 5, [0, 255, 0], -1)

    cv2.imshow('Image', img_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return np.asarray(centers_coordinates)


def debug():
    """
    first show the whole process of manually adjust the corners from the lane used to compute homo
    second show the resultant unwarp image"""
    img_path = "./data/0000000000.png"
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    homo = homoToResCenter(img_gray.shape)

    img_unwarped = cv2.warpPerspective(img, homo, (img_gray.shape[0], img_gray.shape[1]))

    # img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    cv2.imshow('Image unwarped', img_unwarped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def homo():
    return np.array([[-3.52198203e-01, -1.26771921e+00, 5.24205004e+02],
                     [4.33025034e-16, -4.12125334e+00, 1.07911878e+03],
                     [5.80470942e-19, -4.42069468e-03, 1.00000000e+00]])

# eyeballLaneCorners()
debug()
