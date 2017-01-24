from __future__ import division
import numpy as np
from skimage import morphology as skimorph
from scipy.ndimage import gaussian_laplace
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.morphology import binary_opening
from skimage import filters as skifilter
from scipy.ndimage.morphology import binary_erosion
from skimage.segmentation import clear_border
try:
    from covertrack.utils.seg_utils import skilabel, peak_local_max_edge
    from covertrack.utils.seg_utils import calc_neck_score_thres, labels2outlines, cut_neck
except:
    from utils.seg_utils import skilabel, peak_local_max_edge
    from utils.seg_utils import calc_neck_score_thres, labels2outlines, cut_neck
from segment_utils.filters import sizefilterandopen, sizefilter_for_label
from segment_utils.filters import devide_and_label_objects, highpassfilter
from segment_utils.filters import remove_thin_objects, sitk_watershed_intensity
from segment_utils.filters import lap_local_max, extract_foreground_adaptive, calc_lapgauss
from segment_utils.filters import enhance_edges
from skimage.measure import regionprops

np.random.seed(0)


def example_thres(img, holder, THRES=100):
    """take pixel above THRES as a foreground.

    Examples:
        >>> img = np.zeros((3, 3))
        >>> img[0, 0] = 10
        >>> img[2, 2] = 10
        >>> example_thres(img, None, THRES=5)
        array([[1, 0, 0],
               [0, 0, 0],
               [0, 0, 2]])
    """
    return skilabel(img > THRES)


def global_otsu(img, holder, FILTERSIZE=1, DEBRISAREA=50, MAXSIZE=1000,
                OPENING=2, SHRINK=0, REGWSHED=10):
    img = gaussian_filter(img, FILTERSIZE)
    global_thresh = skifilter.threshold_otsu(img)
    bw = img > global_thresh
    bw = sizefilterandopen(bw, DEBRISAREA, MAXSIZE, OPENING)
    if SHRINK > 0:
        bw = binary_erosion(bw, np.ones((SHRINK, SHRINK)))
    label = devide_and_label_objects(bw, REGWSHED)
    label = sizefilter_for_label(label, DEBRISAREA, MAXSIZE, OPENING)
    label = skilabel(clear_border(label, buffer_size=2))
    return label


def logglobal(img, holder, DEBRISAREA=50, MAXSIZE=1000, OPENING=2,
              NUCRAD=10, magnitude=2, SHRINK=0, REGWSHED=10, HPASS=2.5, GLAP=3):
    logimg = np.log(img)
    highpassImg = highpassfilter(logimg, NUCRAD*HPASS)
    sharpLogimg = logimg + magnitude*highpassImg
    lapSharpLogimg = gaussian_laplace(sharpLogimg, NUCRAD/GLAP)
    lapSharpLogimg = lapSharpLogimg - lapSharpLogimg.min()
    lapSharpLogimg = lapSharpLogimg.max() - lapSharpLogimg
    img = lapSharpLogimg
    global_thresh = skifilter.threshold_otsu(img)
    bw = img > global_thresh
    bw = sizefilterandopen(bw, DEBRISAREA, MAXSIZE, OPENING)
    if SHRINK > 0:
        bw = binary_erosion(bw, np.ones((SHRINK, SHRINK)))
    label = devide_and_label_objects(bw, REGWSHED)
    label = sizefilter_for_label(label, DEBRISAREA, MAXSIZE, OPENING)
    label = skilabel(clear_border(label, buffer_size=2))
    return label


def logadaptivegauss(img, holder, DEBRISAREA=50, MAXSIZE=1000, OPENING=2, magnitude=2,
                     NUCRAD=10, FILTERINGSIZE=100, T=10, SHRINK=0, REGWSHED=10, GLAP=3, HPASS=2.5):
    logimg = np.log(img)
    highpassImg = highpassfilter(logimg, NUCRAD*HPASS)
    sharpLogimg = logimg + magnitude*highpassImg
    lapSharpLogimg = gaussian_laplace(sharpLogimg, NUCRAD/GLAP)
    lapSharpLogimg = lapSharpLogimg - lapSharpLogimg.min()
    lapSharpLogimg = lapSharpLogimg.max() - lapSharpLogimg
    img = lapSharpLogimg

    fim = gaussian_filter(img, FILTERINGSIZE)
    bw = img > fim*(1.+T/100.)
    bw = sizefilterandopen(bw, DEBRISAREA, MAXSIZE, OPENING)
    if SHRINK > 0:
        bw = binary_erosion(bw, np.ones((SHRINK, SHRINK)))
    label = devide_and_label_objects(bw, REGWSHED)
    label = sizefilter_for_label(label, DEBRISAREA, MAXSIZE, OPENING)
    label = skilabel(clear_border(label, buffer_size=2))
    return label


def adaptivethresh2blocks(img, holder, ADAPTIVEBLOCK=21, DEBRISAREA=50, MAXSIZE=1000,
                          OPENING=2, FILTERSIZE=1, SHRINK=0, REGWSHED=10):
    img = gaussian_filter(img, FILTERSIZE)
    bw = skifilter.threshold_adaptive(img, ADAPTIVEBLOCK, 'gaussian')
    bw2 = skifilter.threshold_adaptive(img, img.shape[0]/4, 'gaussian')
    bw = bw * bw2
    bw = sizefilterandopen(bw, DEBRISAREA, MAXSIZE, OPENING)
    if SHRINK > 0:
        bw = binary_erosion(bw, np.ones((SHRINK, SHRINK)))
    label = devide_and_label_objects(bw, REGWSHED)
    label = sizefilter_for_label(label, DEBRISAREA, MAXSIZE, OPENING)
    label = skilabel(clear_border(label, buffer_size=2))
    return label


def adaptivethreshwithglobal(img, holder, ADAPTIVEBLOCK=21, DEBRISAREA=50, MAXSIZE=1000,
                             OPENING=2, FILTERSIZE=1, THRESHRELATIVE=1, SHRINK=0, REGWSHED=10):
    img = gaussian_filter(img, FILTERSIZE)
    global_thresh = skifilter.threshold_otsu(img)
    bw = skifilter.threshold_adaptive(img, ADAPTIVEBLOCK, 'gaussian')
    bw = bw * img > global_thresh * THRESHRELATIVE
    bw = sizefilterandopen(bw, DEBRISAREA, MAXSIZE, OPENING)
    if SHRINK > 0:
        bw = binary_erosion(bw, np.ones((SHRINK, SHRINK)))
    label = devide_and_label_objects(bw, REGWSHED)
    label = sizefilter_for_label(label, DEBRISAREA, MAXSIZE, OPENING)
    label = skilabel(clear_border(label, buffer_size=2))
    return label

def constant_lap_edge(img, holder, DEBRISAREA=50, MAXSIZE=1000, OPENING=2, NUCRAD=6, MAGNITUDE=2,
                      SHRINK=0, REGWSHED=10, FILTERSIZE=1, THRES=4, HPASS=8):
    img = gaussian_filter(img, FILTERSIZE)
    edge = enhance_edges(img, HPASS, NUCRAD)
    logimg = np.log(img) - edge * MAGNITUDE
    bw = logimg > THRES
    bw = binary_fill_holes(bw)
    if SHRINK > 0:
        bw = binary_erosion(bw, np.ones((SHRINK, SHRINK)))
    if OPENING != 0: # added by KL
        bw = binary_opening(bw, np.ones((OPENING, OPENING)), iterations=1) #added by KL
    label = devide_and_label_objects(bw, REGWSHED)
    label = sizefilter_for_label(label, DEBRISAREA, MAXSIZE, OPENING)
    label = skilabel(clear_border(label, buffer_size=2))
    return label


def lap_waterhsed_intensity(img, holder, DEBRISAREA=50, BLUR=2, MAXSIZE=5000, OPENING=0,THRES=0.01,
                            SEPARATE=3, FILTERINGSIZE=20, RATIO=1, MIN_SIGMA=8, MAX_SIGMA=12):
    '''
    Approximate sigma is given by sigma=radius/sqrt(2).
    MIN_SIGMA, MAX_SIGMA, SEPARATE are important to deal with oversegmentation...
    '''
    img = gaussian_filter(img, BLUR)
    foreground = extract_foreground_adaptive(img, RATIO, FILTERINGSIZE)
    sigma_list = range(MIN_SIGMA, MAX_SIGMA)
    local_maxima = lap_local_max(img, sigma_list, THRES)
    local_maxima[-foreground] = 0
    local_maxima = skilabel(peak_local_max_edge(local_maxima, SEPARATE))
    label = sitk_watershed_intensity(img, local_maxima)
    label = sizefilter_for_label(label, DEBRISAREA, MAXSIZE, OPENING)
    label = skilabel(clear_border(label, buffer_size=2))
    return label


def adaptivethreshwithglobal_neckcut(img, holder, ADAPTIVEBLOCK=21, DEBRISAREA=50, MAXSIZE=1000,
                                     OPENING=2, FILTERSIZE=1, THRESHRELATIVE=1, SHRINK=0,
                                     REGWSHED=10, THRES_ANGLE=181, EDGELEN=5):
    img = gaussian_filter(img, FILTERSIZE)
    global_thresh = skifilter.threshold_otsu(img)
    bw = skifilter.threshold_adaptive(img, ADAPTIVEBLOCK, 'gaussian')
    bw = bw * img > global_thresh * THRESHRELATIVE
    bw = sizefilterandopen(bw, DEBRISAREA, MAXSIZE, OPENING)
    if SHRINK > 0:
        bw = binary_erosion(bw, np.ones((SHRINK, SHRINK)))
    label = devide_and_label_objects(bw, REGWSHED)

    cl_label = clear_border(label)
    outlines = labels2outlines(cl_label)
    rps = regionprops(outlines)
    good_coords = []
    for cell in rps:
        score, coords = calc_neck_score_thres(cell.coords, edgelen=EDGELEN, thres=THRES_ANGLE)

        if len(score) > 1:
            r0, c0 = coords[0, :]
            # Find candidates
            for cand in coords[1:, :]:
                cut_label = cut_neck(cl_label == cell.label, r0, c0, cand[0], cand[1])
                cand_label = skilabel(cut_label)
                cand_rps = regionprops(cand_label)
                cand_rps = [i for i in cand_rps if i.area > DEBRISAREA]
                if len(cand_rps) > 1:
                    good_coords.append([r0, c0, cand[0], cand[1]])
                    break
        # Reflect cut in the label
        for gc in good_coords:
            cut_neck(label, *gc)
    return label


def lapgauss_adaptive(img, holder, RATIO=3.0, FILTERINGSIZE=50, SIGMA=2.5, DEBRISAREA=50,
                      MAXSIZE=1000, OPENING=2,
                      SHRINK=0, REGWSHED=10, COPEN=1, THINERODE=4):
    bw = extract_foreground_adaptive(img, RATIO, FILTERINGSIZE)
    cimg = calc_lapgauss(img, SIGMA)
    bw[cimg > 0] = 0
    bw = binary_fill_holes(bw)
    bw = skimorph.remove_small_objects(bw, DEBRISAREA, connectivity=4)
    bw = binary_opening(bw, np.ones((COPEN, COPEN)))
    bw = remove_thin_objects(bw, THINERODE)
    bw = sizefilterandopen(bw, DEBRISAREA, MAXSIZE, OPENING)
    if SHRINK > 0:
        bw = binary_erosion(bw, np.ones((SHRINK, SHRINK)))
    label = devide_and_label_objects(bw, REGWSHED)
    label = sizefilter_for_label(label, DEBRISAREA, MAXSIZE, OPENING)
    label = skilabel(clear_border(label, buffer_size=2))
    return label


def lapgauss_constant(img, holder, SIGMA=2.5, DEBRISAREA=50, MAXSIZE=1000, OPENING=2,
                      SHRINK=0, REGWSHED=10, THRES=0.3, COPEN=1, THINERODE=4):
    cimg = calc_lapgauss(img, SIGMA)
    bw = cimg > THRES
    bw = binary_fill_holes(bw)
    bw = skimorph.remove_small_objects(bw, DEBRISAREA, connectivity=4)
    bw = binary_opening(bw, np.ones((COPEN, COPEN)))
    bw = remove_thin_objects(bw, THINERODE)
    bw = sizefilterandopen(bw, DEBRISAREA, MAXSIZE, OPENING)
    if SHRINK > 0:
        bw = binary_erosion(bw, np.ones((SHRINK, SHRINK)))
    label = devide_and_label_objects(bw, REGWSHED)
    label = sizefilter_for_label(label, DEBRISAREA, MAXSIZE, OPENING)
    label = skilabel(clear_border(label, buffer_size=2))
    return label
