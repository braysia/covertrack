import numpy as np
from covertrack.cell import Cell


class CellOperator(object):
    def __init__(self, params):
        self.params = params

def _distance_diff(obj1, obj2):
    xprev = [i.corr_x for i in obj1]
    yprev = [i.corr_y for i in obj1]
    xcurr = [i.corr_x for i in obj2]
    ycurr = [i.corr_y for i in obj2]
    xprevTile = np.tile(xprev, (len(xcurr), 1))
    yprevTile = np.tile(yprev, (len(ycurr), 1))
    return abs(xprevTile.T - xcurr) + abs(yprevTile.T - ycurr)


def _totalintensity_difference(obj1, obj2):
    massprev = [i.total_intensity for i in obj1]
    masscurr = [i.total_intensity for i in obj2]
    massprevTile = np.tile(massprev, (len(masscurr), 1))
    return (masscurr-massprevTile.T)/massprevTile.T
