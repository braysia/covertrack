from __future__ import division
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
from skimage import morphology as skimorph
from scipy.ndimage.morphology import binary_opening, binary_dilation
from skimage.morphology import reconstruction
from pymorph import neg
try:
    from covertrack.utils.seg_utils import watershed, skilabel
except:
    from utils.seg_utils import watershed, skilabel
from scipy.ndimage import gaussian_laplace
from scipy.ndimage.filters import gaussian_filter
from skimage.measure import regionprops
from scipy.ndimage import grey_erosion
import SimpleITK as sitk
from skimage.feature import peak_local_max


def sizefilterandopen(bw, DEBRISAREA, MAXSIZE, OPENING):
    bw = binary_fill_holes(bw)
    bw = skimorph.remove_small_objects(bw, DEBRISAREA, connectivity=4)
    antibw = skimorph.remove_small_objects(bw, MAXSIZE, connectivity=4)
    bw[antibw] = False
    if OPENING != 0:
        bw = binary_opening(bw, np.ones((OPENING, OPENING)), iterations=1)
    return bw


def gray_fill_holes(label):
    '''This will fill holes of gray int images'''
    label = np.int32(label)
    label = np.pad(label, pad_width=1, mode='constant', constant_values=-100000)
    blabel = label.copy()
    blabel[1:-1, 1:-1] = 100000
    fim = reconstruction(neg(blabel), neg(label))
    fim = neg(np.int32(fim))
    fim = fim[1:-1, 1:-1]
    return fim


def devide_and_label_objects(bw, REGWSHED):
    label = skilabel(bw).astype(np.int64)
    if REGWSHED > 0:
        label = watershed(label, REGWSHED)
    return label


def highpassfilter(img, block):
    highpassed = img - gaussian_filter(img, block)
    return highpassed

def remove_thin_objects(bw, THINERODE):
    label = skilabel(bw)
    erod_label = grey_erosion(label, THINERODE)
    unq_label = np.unique(label)
    unq_erod_label = np.unique(erod_label)
    thinobj_id = [i for i in unq_label if i not in unq_erod_label]
    for i in thinobj_id:
        bw[label == i] = 0
    return bw

def simple_highpassfilter(img, block):
    highpassed = img*2 - gaussian_filter(img, block)
    highpassed[highpassed <= 0] = 1
    return highpassed


def enhance_edges(img, HPASS, NUCRAD):
    img = simple_highpassfilter(img.astype(np.float64), HPASS)
    lapgauss_img = -gaussian_laplace(img.astype(np.float32), NUCRAD/3)*(NUCRAD/3)**2
    edge = -lapgauss_img
    return edge


def curvature_anisotropic_smooth(img, NUMITER=10):
    fil = sitk.CurvatureAnisotropicDiffusionImageFilter()
    fil.SetNumberOfIterations(NUMITER)
    simg = sitk.GetImageFromArray(img.astype(np.float32))
    sres = fil.Execute(simg)
    return sitk.GetArrayFromImage(sres)

def calc_lapgauss(img, SIGMA=2.5):
    fil = sitk.LaplacianRecursiveGaussianImageFilter()
    fil.SetSigma(SIGMA)
    # fil.SetNormalizeAcrossScale(False)
    csimg = sitk.GetImageFromArray(img)
    slap = fil.Execute(csimg)
    return sitk.GetArrayFromImage(slap)


def extract_foreground_adaptive(img, RATIO=3.0, FILTERINGSIZE=50):
    bw = adaptive_thresh(img, RATIO, FILTERINGSIZE)
    bw = binary_opening(bw, np.ones((3, 3)))
    bw = binary_dilation(bw, np.ones((5, 5)))
    bw = binary_fill_holes(bw)
    return bw


def adaptive_thresh(img, RATIO=3.0, FILTERINGSIZE=50):
    """Segment as a foreground if pixel is higher than ratio * blurred image.
    If you set ratio 3.0, it will pick the pixels 300 percent brighter than the blurred image.
    """
    fim = gaussian_filter(img, FILTERINGSIZE)
    bw = img > fim * RATIO
    return bw


def sizefilter_for_label(label, DEBRISAREA, MAXSIZE, OPENING):
    """Tyical routines including filling holes, remove small and large objects.
    """
    label = gray_fill_holes(label)
    label = skimorph.remove_small_objects(label, DEBRISAREA, connectivity=4)
    antibw = skimorph.remove_small_objects(label, MAXSIZE, connectivity=4)
    antibw = antibw.astype(bool)
    label[antibw] = 0
    return label


def lap_local_max(img, sigma_list, THRES):
    img = np.uint16(img)
    lapimages = []
    for sig in sigma_list:
        simg = sitk.GetImageFromArray(img)
        nimg = sitk.LaplacianRecursiveGaussian(image1=simg, sigma=sig)
        lapimages.append(-sitk.GetArrayFromImage(nimg))

    image_cube = np.dstack(lapimages)
    local_maxima = peak_local_max(image_cube, threshold_abs=THRES, footprint=np.ones((3, 3, 3)),threshold_rel=0.0,exclude_border=False, indices=False)

    local_maxima = local_maxima.sum(axis=2)
    local_maxima = skilabel(local_maxima)
    return local_maxima


def sitk_watershed_intensity(img, local_maxima):
    seedimage = sitk.GetImageFromArray(local_maxima.astype(np.uint16))#

    img = img.astype(np.float32)
    nimg = sitk.GetImageFromArray(img)
    nimg = sitk.GradientMagnitude(nimg)#

    fil = sitk.MorphologicalWatershedFromMarkersImageFilter()
    fil.FullyConnectedOn()
    fil.MarkWatershedLineOff()
    oimg1 = fil.Execute(nimg, seedimage)
    labelim = sitk.GetArrayFromImage(oimg1)
    return labelim


def clean_sitk_watershed_intensity(labelim, bw, DEBRISAREA):
    # Background is not necessary to be 0, so repair the labelim.
    back = labelim.copy()
    back[bw] = 0
    unique_pix = np.unique(back)
    for up in unique_pix:
        labelim[labelim == up] = 0
    regions = regionprops(labelim)
    for region in regions:
        if region.area < DEBRISAREA:
            labelim[labelim == region.label] = 0
    labelim = sitk.GetArrayFromImage(sitk.GrayscaleFillhole(sitk.GetImageFromArray(labelim)))
    return labelim
