import numpy as np
from __main__ import *
from HistorgramOfLanePixels import *


def fitThemLines(x, y, n,allpoints):

    lines=np.polyfit(y,x, 2)
    lines=np.poly1d(lines)



    xpoints=lines(allpoints)


    lines=np.array([xpoints,allpoints],np.int32).T



    return lines


def MarkLanes(bincntframe,flatBGR,frame):
    bincntframeTop = bincntframe[:int(bincntframe.shape[0] / 3), :]
    bincntframeMid = bincntframe[int(bincntframe.shape[0] / 3):int(bincntframe.shape[0] * 2 / 3), :]
    bincntframeBot = bincntframe[int(bincntframe.shape[0] * 2 / 3):, :]

    ypoints = np.array([int((0 + int(bincntframe.shape[0] / 3)) / 2),
                        int((int(bincntframe.shape[0] / 3) + int(bincntframe.shape[0] * 2 / 3)) / 2),
                        int((int(bincntframe.shape[0] * 2 / 3) + bincntframe.shape[0]) / 2)]).T

    LeftT, RightT = findLeftAndRightPoints(bincntframeTop)
    LeftM, RightM = findLeftAndRightPoints(bincntframeTop)
    LeftB, RightB = findLeftAndRightPoints(bincntframeTop)

    Xright = np.array([RightT, RightM, RightB]).T
    Xleft = np.array([LeftT, LeftM, LeftB]).T

    wholeframe = np.arange(0, frame.shape[1])

    Leftlines = fitThemLines(Xleft, ypoints, 3, wholeframe)

    Rightlines = fitThemLines(Xright, ypoints, 3, wholeframe)

    flatBGR = cv2.polylines(flatBGR, [Leftlines], True, (0, 0, 255), 5)
    MarkedLanes = cv2.polylines(flatBGR, [Rightlines], True, (255, 0, 0), 5)


    return MarkedLanes,Leftlines,Rightlines