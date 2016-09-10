import numpy as np
from os.path import exists
from os.path import join
import os
from scipy.ndimage.morphology import grey_opening
from skimage.transform import resize
from scipy.ndimage.filters import median_filter
from covertrack.utils.seg_utils import adaptive_thresh
from preprocess_utils.preprocess_utils import homogenize_intensity_n4, wavelet_subtraction
from preprocess_utils.preprocess_utils import convert_positive, estimate_background_prc
from preprocess_utils.preprocess_utils import curvature_anisotropic_smooth, resize_img
from preprocess_utils.preprocess_utils import histogram_matching, wavelet_subtraction_hazen
import SimpleITK as sitk


def hist_matching(img, holder, BINS=10000, QUANT=100):
    """Histogram matching.
    """
    if not hasattr(holder, 'prev_img'):
        holder.prev_img = img
    else:
        img = histogram_matching(img, holder.prev_img, BINS, QUANT)
        holder.prev_img = img
    return img


def n4_illum_correction(img, holder, RATIO=1.5, FILTERINGSIZE=50):
    """
    Implementation of the N4 bias field correction algorithm.
    Takes some calculation time. It first calculates the background using adaptive_thesh.
    """
    bw = adaptive_thresh(img, RATIO=RATIO, FILTERINGSIZE=FILTERINGSIZE)
    img = homogenize_intensity_n4(img, -bw)
    return img


def n4_illum_correction_downsample(img, holder, DOWN=2, RATIO=1.05, FILTERINGSIZE=50, OFFSET=10):
    """Faster but more insensitive to local illum bias.
    """
    fil = sitk.ShrinkImageFilter()
    cc = sitk.GetArrayFromImage(fil.Execute(sitk.GetImageFromArray(img), [DOWN, DOWN]))
    bw = adaptive_thresh(cc, RATIO=RATIO, FILTERINGSIZE=FILTERINGSIZE/DOWN)
    himg = homogenize_intensity_n4(cc, -bw)
    himg = cc - himg
    # himg[himg < 0] = 0
    bias = resize_img(himg, img.shape)
    img = img - bias
    return convert_positive(img, OFFSET)


def background_subtraction_wavelet(img, holder, level=7, OFFSET=10):
    '''
    It might be radical but works in many cases in terms of segmentation.
    Use "background_subtraction_wavelet_hazen" for a proper implementation.
    '''
    img = wavelet_subtraction(img, level)
    return convert_positive(img, OFFSET)


def smooth_curvature_anisotropic(img, holder, NUMITER=10):
    """anisotropic diffusion on a scalar using the modified curvature diffusion equation (MCDE).
    """
    return curvature_anisotropic_smooth(img, NUMITER)


def background_subtraction_prcblock(img, holder, BLOCK=10, OPEN=3, PERCENTILE=0.1, OFFSET=50):
    img = median_filter(img, size=(5, 5))
    img = estimate_background_prc(img, BLOCK, PERCENTILE)
    img = convert_positive(img, OFFSET)
    return img

def background_subtraction_wavelet_hazen(img, holder, THRES=100, ITER=5, WLEVEL=6, OFFSET=50):
    """Wavelet background subtraction for STORM.
    """
    back = wavelet_subtraction_hazen(img, ITER=ITER, THRES=THRES, WLEVEL=WLEVEL)
    img = img - back
    return convert_positive(img, OFFSET)
