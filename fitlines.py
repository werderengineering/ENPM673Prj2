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

    Tweight=0
    Mweight=0
    Bweight=0
    bincntframeTop = bincntframe[:int(bincntframe.shape[0] / 3), :]
    bincntframeMid = bincntframe[int(bincntframe.shape[0] / 3):int(bincntframe.shape[0] * 2 / 3), :]
    bincntframeBot = bincntframe[int(bincntframe.shape[0] * 2 / 3):, :]

    ypoints = np.array([int((0 + int(bincntframe.shape[0] / 3)) / 2),
                        int((int(bincntframe.shape[0] / 3) + int(bincntframe.shape[0] * 2 / 3)) / 2),
                        int((int(bincntframe.shape[0] * 2 / 3) + bincntframe.shape[0]) / 2)]).T

    LeftTotal, RightTotal = findLeftAndRightPoints(bincntframe)

    LeftT, RightT = findLeftAndRightPoints(bincntframeTop)
    LeftM, RightM = findLeftAndRightPoints(bincntframeMid)
    LeftB, RightB = findLeftAndRightPoints(bincntframeBot)

    Xright = np.array([RightT * Tweight, RightM * Mweight, RightB * Bweight])
    Xleft = np.array([LeftT * Tweight, LeftM * Mweight, LeftB * Bweight])
    #
    # Xright = (((np.array([RightT*Tweight, RightM*Mweight, RightB*Bweight])-RightTotal)*.1)+RightTotal).T.astype(int)
    # Xleft = (((np.array([LeftT*Tweight, LeftM*Mweight, LeftB*Bweight])-LeftTotal)*.1)+LeftTotal).T.astype(int)

    MidTop=RightT-LeftT
    MidBot=RightB-LeftB

    Turning=MidTop-MidBot

    wholeframe = np.arange(0, frame.shape[1])

    Leftlines = fitThemLines(Xleft, ypoints, 3, wholeframe)

    Rightlines = fitThemLines(Xright, ypoints, 3, wholeframe)

    flatBGR = cv2.polylines(flatBGR, [Leftlines], False, (0, 0, 255), 5)
    MarkedLanes = cv2.polylines(flatBGR, [Rightlines], False, (255, 0, 0), 5)


    return MarkedLanes,Leftlines,Rightlines,Turning