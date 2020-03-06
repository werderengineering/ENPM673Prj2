import numpy as np
from __main__ import *
from HistorgramOfLanePixels import *


def fitThemLines(x, y, n,allpoints):

    lines=np.polyfit(y,x, 4,True)
    lines=np.poly1d(lines)



    xpoints=lines(allpoints)



    lines=np.array([xpoints,allpoints],np.int32).T



    return lines, xpoints


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

    LeftT, RightT = findLeftAndRightPoints(bincntframeTop)
    LeftM, RightM = findLeftAndRightPoints(bincntframeMid)
    LeftB, RightB = findLeftAndRightPoints(bincntframeBot)
    ######################################################################################################

    bincntframeTH = bincntframe[:int(bincntframe.shape[0] / 2), :]
    bincntframeBH = bincntframe[int(bincntframe.shape[0] / 2):, :]
    LeftTH, RightTH = findLeftAndRightPoints(bincntframeTH)
    LeftBH, RightBH = findLeftAndRightPoints(bincntframeBH)
    ######################################################################################################



    LeftTotal, RightTotal = findLeftAndRightPoints(bincntframe)
    ######################################################################################################


    Xright = np.array([RightTotal, RightTotal, RightTotal])
    Xleft = np.array([LeftTotal, LeftTotal, LeftTotal])
    # ypoints = np.array([int((0 + int(bincntframe.shape[0]/2)) / 2),                   int((int(bincntframe.shape[0]) + bincntframe.shape[0]) / 2)]).T
    ######################################################################################################



    # Xright = np.array([RightTH, RightBH])
    # Xleft = np.array([LeftTH, LeftBH])



    wholeframe = np.arange(0, frame.shape[1])

    Leftlines, LeftDiferential = fitThemLines(Xleft, ypoints, 3, wholeframe)

    Rightlines, RightDiferential = fitThemLines(Xright, ypoints, 3, wholeframe)

    Turning =(RightDiferential-LeftDiferential)

    # LeftTop=LeftDiferential[len(LeftDiferential)-1]
    # RightTop=RightDiferential[len(RightDiferential)-1]
    # LeftBot = LeftDiferential[0]
    # RightBot = RightDiferential[0]
    #
    #
    # TopDiff=RightTop-LeftTop
    # BotDiff=RightBot-LeftBot
    #
    # if abs(TopDiff-BotDiff)<=2:
    #     print('Straight')
    #
    # elif TopDiff>BotDiff:
    #     print('Turning Right')
    #
    # elif TopDiff<BotDiff:
    #     print('Turning Left')
    # print('\nTop',TopDiff)
    # print('Bottom', BotDiff)
    # print('\n')
    # print('Turning difference',Turning)
    LeftlinesM=Leftlines[200:1100]
    RightlinesM = Rightlines[200:1100]

    flatBGR = cv2.polylines(flatBGR, [LeftlinesM], False, (0, 0, 255), 5)
    MarkedLanes = cv2.polylines(flatBGR, [RightlinesM], False, (255, 0, 0), 5)


    return MarkedLanes,Leftlines,Rightlines, Turning


def CheckTurn(Right,Left):
    Right=Right[:,0]
    Left=Left[:,0]

    LeftBot = Left[len(Left) - 200]
    RightBot = Right[len(Right) - 200]
    LeftTop = Left[0]
    RightTop = Right[0]

    TopDiff = RightTop - LeftTop
    BotDiff = RightBot - LeftBot

    if abs(TopDiff - BotDiff) <= 2:
        print('Straight')
        Turning=0

    elif TopDiff > BotDiff:
        print('Turning Right')
        Turning=1

    elif TopDiff < BotDiff:
        print('Turning Left')
        Turning=-1

    print('\nTop', TopDiff)
    print('Bottom', BotDiff)
    print('\n')
    return Turning