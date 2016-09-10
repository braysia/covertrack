import numpy as np
from skimage.measure import regionprops
from copy import deepcopy

PROP_SAVE = ['area', 'cell_id', 'convex_area', 'corr_x', 'corr_y', 'cv_intensity',
             'eccentricity', 'equivalent_diameter', 'euler_number', 'extent', 'filled_area',
             'major_axis_length', 'max_intensity', 'mean_intensity',
             'median_intensity', 'min_intensity', 'orientation',
             'perimeter', 'solidity', 'std_intensity', 'total_intensity', 'x', 'y',
             'coords']


class CellListMaker(object):
    '''Make a list of Cell objects'''
    def __init__(self, img, label, params, frame=0):
        self.img = img
        self.label = label
        self.params = params
        self.frame = frame

    def make_list(self):
        cell_prop = regionprops(self.label, self.img, cache=True)
        celllist = [Cell(i, self.frame) for i in cell_prop]
        return celllist


class CellListMakerScalar(CellListMaker):
    '''Make a list of Cell objects but remove any regionprops features
    which are tuple, list or array to reduce memory usage.
    '''
    def make_list(self):
        if self.label.any():
            cell_prop = regionprops(self.label, self.img, cache=True)
            celllist = [Cell(i, self.frame) for i in cell_prop]
            features = [i for i in dir(celllist[0].prop) if not i.startswith('_')]
            fremoved = []
            for i in features:
                if type(getattr(celllist[0].prop, i)) in (tuple, list, np.ndarray):
                    fremoved.append(i)
            for i in fremoved:
                [j.prop.__delattr__(i) for j in celllist]
            return celllist
        else:
            return []


class Prop(object):
    def __init__(self, prop):
        for ki in prop.__class__.__dict__.iterkeys():
            if '__' not in ki:
                setattr(self, ki, prop.__getitem__(ki))
        self.label_id = prop.label
        pix = prop['intensity_image']
        pix = pix[pix != 0]

        # CAUTION
        # This will not reflected to the objects labels (segmentation)
        # if len(pix) > 2:
        #     pix = pix[(pix > np.nanpercentile(pix, 10)) * (pix<np.nanpercentile(pix, 90))]

        self.mean_intensity = np.mean(pix)
        self.median_intensity = np.median(pix)
        self.total_intensity = prop['area'] * np.mean(pix)
        self.std_intensity = np.std(pix)
        self.cv_intensity = np.std(pix)/np.mean(pix)
        self.x = self.centroid[1]
        self.corr_x = self.centroid[1]  # will updated when jitter corrected
        self.y = self.centroid[0]
        self.corr_y = self.centroid[0]  # will updated when jitter corrected
        self.parent_id = 0
        self.frame = np.nan
        self.abs_id = 0
        self.cell_id = 0

class PropLight(object):
    def __init__(self, prop):
        for ki in prop.__class__.__dict__.iterkeys():
            if ki in PROP_SAVE:
                setattr(self, ki, prop.__getitem__(ki))
        self.label_id = prop.label
        pix = prop['intensity_image']
        pix = pix[pix != 0]

        # CAUTION
        # This will not reflected to the objects labels (segmentation)
        # if len(pix) > 2:
        #     pix = pix[(pix > np.nanpercentile(pix, 10)) * (pix<np.nanpercentile(pix, 90))]
        self.mean_intensity = np.mean(pix, dtype=np.float32)
        self.median_intensity = np.median(pix)
        self.total_intensity = prop['area'] * np.mean(pix, dtype=np.float32)
        self.std_intensity = np.std(pix, dtype=np.float32)
        self.cv_intensity = np.std(pix, dtype=np.float32)/np.mean(pix, dtype=np.float32)
        self.x = prop['centroid'][1]
        self.corr_x = prop['centroid'][1]  # will updated when jitter corrected
        self.y = prop['centroid'][0]
        self.corr_y = prop['centroid'][0]  # will updated when jitter corrected
        self.parent_id = 0
        self.frame = np.nan
        self.abs_id = 0
        self.cell_id = 0


class Cell(object):
    '''Cell object which holds Prop.
    self.next and self.previous will return an associated cell in the next
    frame or previous frame if available.
    '''
    def __init__(self, prop, frame):
        self.frame = frame
        self.prop = PropLight(prop)
        self.cell_id = None
        self.parent = None
        self._next = None
        self.previous = None

    @property
    def next(self):
        return self._next

    @next.setter
    def next(self, partner):
        self._next = partner
        partner.previous = self
