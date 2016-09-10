import numpy as np
from covertrack.cell import Cell


class CellOperator(object):
    def __init__(self, params):
        self.params = params


class Container(object):
    '''
    container contains untr_prev, untr_curr, tr_prev, tr_curr and params.
    Once untr_prev and untr_curr are linked, they move to tr_prev and tr_curr.
    '''
    def __init__(self, img_shape, curr_cells=None, prev_cells=None):
        self.curr_cells = curr_cells
        self.prev_cells = prev_cells
        self.img_shape = img_shape

    @property
    def unlinked(self):
        unlinked_prev = [i for i in self.prev_cells if i.next is None]
        unlinked_curr = [i for i in self.curr_cells if i.previous is None]
        return [unlinked_prev, unlinked_curr]

    @property
    def linked(self):
        linked_prev = [i for i in self.prev_cells if i.next is not None]
        linked_curr = [i for i in self.curr_cells if i.previous is not None]
        return [linked_prev, linked_curr]

    @property
    def _dist_unlinked(self):
        distdiff = _distance_diff(*self.unlinked)
        return distdiff

    @property
    def _dist_linked(self):
        distdiff = _distance_diff(*self.linked)
        return distdiff

    @property
    def _masschange_unlinked(self):
        masschange = _totalintensity_difference(*self.unlinked)
        return masschange

    @property
    def _label_tracked(self):
        label_tracked = np.uint16(np.zeros(self.img_shape))
        for ob in self.linked[1]:
            label_tracked[ob.prop.coords[:, 0], ob.prop.coords[:, 1]] = ob.prop.label_id
        return label_tracked

    @property
    def _label_untracked(self):
        label_untracked = np.uint16(np.zeros(self.img_shape))
        for ob in self.unlinked[1]:
            label_untracked[ob.prop.coords[:, 0], ob.prop.coords[:, 1]] = ob.prop.label_id
        return label_untracked

    @property
    def _label(self):
        label = np.logical_or(self.label_tracked(), self.label_untracked())
        return label


def _distance_diff(obj1, obj2):
    xprev = [i.prop.corr_x for i in obj1]
    yprev = [i.prop.corr_y for i in obj1]
    xcurr = [i.prop.corr_x for i in obj2]
    ycurr = [i.prop.corr_y for i in obj2]
    xprevTile = np.tile(xprev, (len(xcurr), 1))
    yprevTile = np.tile(yprev, (len(ycurr), 1))
    return abs(xprevTile.T - xcurr) + abs(yprevTile.T - ycurr)


def _totalintensity_difference(obj1, obj2):
    massprev = [i.prop.total_intensity for i in obj1]
    masscurr = [i.prop.total_intensity for i in obj2]
    massprevTile = np.tile(massprev, (len(masscurr), 1))
    return (masscurr-massprevTile.T)/massprevTile.T
