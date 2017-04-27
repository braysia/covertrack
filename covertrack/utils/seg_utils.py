from __future__ import division
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops
from skimage.measure import label as skim_label
from skimage.morphology import watershed as skiwatershed
from skimage.feature import peak_local_max
from skimage.segmentation import find_boundaries
from skimage.feature import peak_local_max
from scipy.ndimage.filters import maximum_filter
from skimage.draw import line
from scipy.ndimage.filters import gaussian_filter
import math
import SimpleITK as sitk

def watershed(label, regmax):
    # Since there are non-unique values for dist, add very small numbers. This will separate each marker by regmax at least.
    dist = distance_transform_edt(label) + np.random.rand(*label.shape)*1e-10
    labeled_maxima = skilabel(peak_local_max(dist, min_distance=regmax, indices=False))
    wshed = -dist
    wshed = wshed - np.min(dist)
    markers = np.zeros(wshed.shape, np.int16)
    markers[labeled_maxima>0]= -labeled_maxima[labeled_maxima>0]
    wlabel = skiwatershed(wshed, markers, connectivity=np.ones((3,3), bool), mask=label!=0)
    wlabel = -wlabel
    wlabel = label.max() + wlabel
    wlabel[wlabel == label.max()] = 0
    all_label = skilabel(label + wlabel)
    return all_label


def skilabel(bw, conn=2):
    '''original label might label any objects at top left as 1. To get around this pad it first.'''
    bw = np.pad(bw, pad_width=1, mode='constant', constant_values=False)
    label = skim_label(bw, connectivity=conn)
    label = label[1:-1, 1:-1]
    return label

def peak_local_max_edge(label, min_dist=5):
    '''peak_local_max sometimes shows a weird behavior...?'''
    label_max = maximum_filter(label, size=min_dist)
    mask = label == label_max
    label[-mask] = 0
    return label


def find_label_boundaries(label):
    blabel = label.copy()
    bwbound = find_boundaries(blabel)
    blabel[-bwbound] = 0
    return blabel

def labels2outlines(labels):
    """Same functionality with find_label_boundaries.
    """
    outlines = labels.copy()
    outlines[-find_boundaries(labels)] = 0
    return outlines

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def cart2pol_angle(x, y):
    phi = np.arctan2(y, x)
    return phi


def order_coords(coords):
    """Order an array of coordinates into a clockwise"""
    new_x = coords[:, 0] - np.unique(coords[:, 0]).mean()
    new_y = coords[:, 1] - np.unique(coords[:, 1]).mean()
    phis = cart2pol_angle(new_x, new_y)
    return coords[np.flipud(np.argsort(phis)), :]


def calc_clockwise_degree(p, c, q):
    """Return an degree in clockwise if you give three points. c will be a center.
    >>> q = [10, 10]
    >>> c = [0, 0]
    >>> p = [-10, 10]
    >>> calc_closewise_degree(p, c, q)
    90.0
    """
    angle_r = cart2pol_angle(q[0]-c[0], q[1]-c[1]) - cart2pol_angle(p[0]-c[0], p[1]-c[1])
    angle = 180.0 * angle_r/np.pi
    if angle < 0:
        angle += 360.0
    return angle


def calc_neck_score(coords, edgelen=3):
    """Calculate the score (angle changes) and return the sorted score and the corresponding pixel
    coordinates. Pass the coordinates of outlines without border."""
    ordered_c = order_coords(coords)
    nc = np.vstack((ordered_c, ordered_c[:edgelen, :]))
    score = []
    for n, ci in enumerate(nc[:-edgelen, :]):
        score.append(calc_clockwise_degree(ordered_c[n-edgelen, :], nc[n, :], nc[n+edgelen, :]))
    idx = np.flipud(np.argsort(score))
    return np.array(score)[idx], ordered_c[idx]


def calc_neck_score_thres(coords, edgelen=5, thres=180):
    score, s_coords = calc_neck_score(coords, edgelen=edgelen)
    return score[score > thres], s_coords[score > thres]


def calc_neck_score_thres_filtered(coords, edgelen=5, thres=180, steplim=5):
    """If two coordinates are too close, ignore the association.
    If steplim is 5, then two points which can be cut should be separated by more than 5 pixels
    if you track a boundary line.
    """
    ordered_c = order_coords(coords)
    score, s_coords = calc_neck_score_thres(coords, edgelen=edgelen, thres=thres)
    idx = []
    for num, co in enumerate(s_coords):
        min_step = calc_shortest_step_coords(ordered_c, s_coords[0, :], co)
        if min_step >= steplim:
            idx.append(num)
    if idx:
        idx.insert(0, 0)
        return score[np.array(idx)], s_coords[np.array(idx)]
    else:
        return np.array([]), np.array([])


def calc_shortest_step_coords(coords, co1, co2):
    co1 = [i for i, c in enumerate(coords) if (c == co1).all()][0]
    co2 = [i for i, c in enumerate(coords) if (c == co2).all()][0]
    return min(abs(co2 - co1), abs(co1 + coords.shape[0] - co2))


def cut_neck(template, r0, c0, r1, c1):
    """Given a label image and two coordinates, it will draw a line which intensity is 0.
    """
    # rr, cc, _ = line_aa(r0, c0, r1, c1)
    rr, cc = line(r0, c0, r1, c1)
    template[rr, cc] = 0
    return template


def get_cut_neck_label(label, r0, c0, r1, c1, debrisarea):
    cut_label = cut_neck(label, r0, c0, r1, c1)
    new_label = skilabel(cut_label)
    rps = regionprops(new_label)
    rps = [i for i in rps if i.area >= debrisarea]
    if len(rps) <= 1:
        return None
    return rps


def adaptive_thresh(img, RATIO=3.0, FILTERINGSIZE=50):
    """Segment as a foreground if pixel is higher than ratio * blurred image.
    If you set ratio 3.0, it will pick the pixels 300 percent brighter than the blurred image.
    """
    fim = gaussian_filter(img, FILTERINGSIZE)
    bw = img > (fim * RATIO)
    return bw

def circularity_thresh(labelim, circ_thres=0.8):
    regions = regionprops(labelim)
    for region in regions:
        circ = 4*math.pi*(region.area/(region.perimeter**2))
        if circ < circ_thres:
            labelim[labelim == region.label] = 0
    labelim = sitk.GetArrayFromImage(sitk.GrayscaleFillhole(sitk.GetImageFromArray(labelim)))
    return labelim