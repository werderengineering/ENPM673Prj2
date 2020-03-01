import numpy as np
from __main__ import *


def fitThemLines(x, y, n):

    lines=np.polyfit(x, y, n)

    return lines