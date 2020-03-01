import numpy as np
from __main__ import *


def fitThemLines(x, y, n):

    lines=np.polyfit(y,x, 2)
    lines=np.poly1d(lines)

    ypoints=np.linspace(0,1,len(x))

    xpoints=lines(y)

    pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)

    lines=np.array([xpoints,y],np.int32).T



    return lines