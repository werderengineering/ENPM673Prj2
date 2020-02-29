import numpy as np
import perspective
import cv2
import imutils


def homoToResCenter(shape_res):
    shape_res = (np.asarray(shape_res) / 2).astype(int)
    src_corners_coordinates = eyeballLaneCorners()
    dest_corners_coordinates = perspective.square()
    dest_corners_coordinates, _ = perspective.shiftToASpot(dest_corners_coordinates, shape_res)
    homo, _ = cv2.findHomography(src_corners_coordinates, dest_corners_coordinates, 0)
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
    img_path = "./data/0000000000.png"
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    homo = homoToResCenter(img_gray.shape)

    img_unwarped = cv2.warpPerspective(img_gray, homo, (img_gray.shape[0], img_gray.shape[1]))

    # img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    cv2.imshow('Image', img_gray)
    cv2.waitKey(0)
    cv2.imshow('Image unwarped', img_unwarped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# eyeballLaneCorners()
debug()
