import sys
from os.path import abspath, dirname
sys.path.append(dirname(dirname(dirname(abspath(__file__)))))
import numpy as np
import pymorph
from skimage.morphology import opening
from skimage.morphology import disk
import SimpleITK as sitk
from centrosome.propagate import propagate
from skimage.morphology import closing
from scipy.signal import convolve2d
from skimage.morphology import disk
try:
    from covertrack.utils.seg_utils import skilabel
except:
    from utils.seg_utils import skilabel
from scipy.ndimage.filters import gaussian_filter
from skimage.measure import regionprops


def dilate_sitk(label, RAD):
    slabel = sitk.GetImageFromArray(label)
    gd = sitk.GrayscaleDilateImageFilter()
    gd.SetKernelRadius(RAD)
    return sitk.GetArrayFromImage(gd.Execute(slabel))

def dilate_to_cytoring(label, RINGWIDTH, MARGIN):
    """
    Examples:
        >>> template = np.zeros((5, 5))
        >>> template[2, 2] = 1
        >>> dilate_to_cytoring(template, 1, 0)
        array([[0, 0, 0, 0, 0],
               [0, 1, 1, 1, 0],
               [0, 1, 0, 1, 0],
               [0, 1, 1, 1, 0],
               [0, 0, 0, 0, 0]], dtype=uint16)
        >>> dilate_to_cytoring(template, 2, 1)
        array([[0, 1, 1, 1, 0],
               [1, 0, 0, 0, 1],
               [1, 0, 0, 0, 1],
               [1, 0, 0, 0, 1],
               [0, 1, 1, 1, 0]], dtype=uint16)
    """
    dilated_nuc = dilate_sitk(label.astype(np.int32), RINGWIDTH)
    comp_dilated_nuc = 1e4 - label
    comp_dilated_nuc[comp_dilated_nuc == 1e4] = 0
    comp_dilated_nuc = dilate_sitk(comp_dilated_nuc.astype(np.int32), RINGWIDTH)
    comp_dilated_nuc = 1e4 - comp_dilated_nuc
    comp_dilated_nuc[comp_dilated_nuc == 1e4] = 0
    dilated_nuc[comp_dilated_nuc != dilated_nuc] = 0
    if MARGIN == 0:
        antinucmask = label
    else:
        antinucmask = dilate_sitk(np.int32(label), MARGIN)
    dilated_nuc[antinucmask.astype(bool)] = 0
    return dilated_nuc.astype(np.uint16)


def dilate_to_cytoring_buffer(label, RINGWIDTH, MARGIN, BUFFER):
    """
    Examples:
        >>> template = np.zeros((5, 5))
        >>> template[1, 1] = 1
        >>> template[-2, -2] = 2
        >>> dilate_to_cytoring_buffer(template, 2, 0, 1)
        array([[1, 1, 0, 0, 0],
               [1, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 2],
               [0, 0, 0, 2, 2]], dtype=uint16)
    """


    dilated_nuc = dilate_sitk(label.astype(np.int32), RINGWIDTH)
    mask = calc_mask_exclude_overlap(label, RINGWIDTH+BUFFER)
    comp_dilated_nuc = 1e4 - label
    comp_dilated_nuc[comp_dilated_nuc == 1e4] = 0
    comp_dilated_nuc = dilate_sitk(comp_dilated_nuc.astype(np.int32), RINGWIDTH)
    comp_dilated_nuc = 1e4 - comp_dilated_nuc
    comp_dilated_nuc[comp_dilated_nuc == 1e4] = 0
    dilated_nuc[comp_dilated_nuc != dilated_nuc] = 0
    if MARGIN == 0:
        antinucmask = label
    else:
        antinucmask = dilate_sitk(np.int32(label), MARGIN)
    dilated_nuc[antinucmask.astype(bool)] = 0
    dilated_nuc[mask] = 0
    return dilated_nuc.astype(np.uint16)

def double_propagate(img, label, FIR_THRES, SEC_THRES, OPEN):
    thres = img > FIR_THRES
    thres2 = img > SEC_THRES
    open_img = opening(img, np.ones((OPEN, OPEN)))
    img = img - open_img
    labels_out, distances = propagate(img, label, thres, 1)
    label = labels_out
    labels_out, distances = propagate(img, label, thres2, 1)
    labels_out = np.uint16(labels_out)
    return labels_out


def calc_mask_exclude_overlap(nuclabel, RINGWIDTH=5):
    """
    Examples:
        >>> template = np.zeros((5, 5))
        >>> template[1, 1] = 1
        >>> template[-2, -2] = 2
        >>> calc_mask_exclude_overlap(template, 2)
        array([[False, False, False, False, False],
               [False, False,  True,  True, False],
               [False,  True,  True,  True, False],
               [False,  True,  True, False, False],
               [False, False, False, False, False]], dtype=bool)
    """
    dilated_nuc = dilate_sitk(nuclabel.astype(np.int32), RINGWIDTH)
    comp_dilated_nuc = 6e4 - nuclabel
    comp_dilated_nuc[comp_dilated_nuc == 6e4] = 0
    comp_dilated_nuc = dilate_sitk(comp_dilated_nuc.astype(np.int32), RINGWIDTH)
    comp_dilated_nuc = 6e4 - comp_dilated_nuc
    comp_dilated_nuc[comp_dilated_nuc == 6e4] = 0
    mask = comp_dilated_nuc != dilated_nuc
    return mask


def gradient_anisotropic(img):
    fil = sitk.GradientAnisotropicDiffusionImageFilter()
    fil.SetNumberOfIterations(10)
    simg = sitk.GetImageFromArray(img.astype(np.float32))
    sres = fil.Execute(simg)
    return sitk.GetArrayFromImage(sres)

def homogenize_cell_intensity_N4(img, bw):
    simg = sitk.GetImageFromArray(img.astype(np.float32))
    sbw = sitk.GetImageFromArray((bw).astype(np.uint8))
    fil = sitk.N4BiasFieldCorrectionImageFilter()
    cimg = fil.Execute(simg, sbw)
    return sitk.GetArrayFromImage(cimg)

def propagate_and_cleanup(img, nuclabel, all_label, mask, prop_param=1.0, nuc_overlap=4):
    label = all_label.copy()
    label[all_label>0] = nuclabel.max() + 1
    label[nuclabel>0] = 0
    label += nuclabel

    labels_out, distances = propagate(img, label, mask, prop_param)
    mask = calc_mask_exclude_overlap(all_label, nuc_overlap)
    labels_out[mask] = 0
    labels_out[labels_out==nuclabel.max()] = 0
    mask = calc_mask_exclude_overlap(labels_out, 1)
    labels_out[mask] = 0
    return labels_out


def label_nearest(img, label, nuc):
    """Label objects to the nearest nuc.
    """
    nuc_prop = regionprops(nuc, img, cache=False)
    sal_prop = regionprops(label, img, cache=False)
    nuc_loc = [i.centroid for i in regionprops(nuc, img, cache=False)]
    sal_loc = [i.centroid for i in regionprops(label, img, cache=False)]
    dist = pairwise_distance(nuc_loc, sal_loc)
    min_dist_arg = np.argmin(dist, axis=0)
    template = np.zeros(img.shape)
    for idx, sal in zip(min_dist_arg, sal_prop):
        if 1:  # if too far remove it?
            template[sal.coords[:, 0], sal.coords[:, 1]] = nuc_prop[idx].label
    comb = np.max(np.dstack((template, nuc)), axis=2).astype(np.uint16)
    return template, comb, nuc_prop, nuc_loc


def pairwise_distance(loc1, loc2):
    xprev = [i[0] for i in loc1]
    yprev = [i[1] for i in loc1]
    xcurr = [i[0] for i in loc2]
    ycurr = [i[1] for i in loc2]
    xprevTile = np.tile(xprev, (len(xcurr), 1))
    yprevTile = np.tile(yprev, (len(ycurr), 1))
    return abs(xprevTile.T - xcurr) + abs(yprevTile.T - ycurr)



def calc_high_pass_kernel(slen, SIGMA):
    """For Salmonella"""
    temp = np.zeros((slen, slen))
    temp[int(slen/2), int(slen/2)] = 1
    gf = gaussian_filter(temp, SIGMA)
    norm = np.ones((slen, slen))/(slen**2)
    return gf - norm

def calc_high_pass(img, slen, SIGMA):
    """For Salmonella"""
    kernel = calc_high_pass_kernel(slen, SIGMA)
    cc = convolve2d(img, kernel, mode='same')
    return cc

def label_high_pass(img, slen=3, SIGMA=0.5, THRES=50, CLOSE=3):
    """For Salmonella"""
    cc = calc_high_pass(img, slen, SIGMA)
    cc[cc < 0] = 0
    la = skilabel(cc > THRES, conn=1)
    la = closing(la, disk(CLOSE))
    return la


def judge_bad(csig, psig, THRESCHANGE):
    if csig - psig > THRESCHANGE:
        return True
    else:
        return False


def repair_sal(img, pimg, comb, pcomb, label, nuc_prop, nuc_loc, THRESCHANGE=1000):
    # repair
    prev = regionprops(pcomb, pimg, cache=False)
    prev_label = [i.label for i in prev]
    curr = regionprops(comb, img, cache=False)
    curr_label = [i.label for i in curr]

    store = []
    for cell in curr:
        curr_sig = cell.mean_intensity * cell.area
        if cell.label in prev_label:
            p_cell = prev[prev_label.index(cell.label)]
            prev_sig = p_cell.mean_intensity * p_cell.area
        else:
            break
        if np.any(label == cell.label):
            store.append(curr_sig - prev_sig)
        if judge_bad(curr_sig, prev_sig, THRESCHANGE):  # or diff ratio?
            for rp in regionprops(skilabel(label == cell.label), img, cache=False):
                dist = pairwise_distance((rp.centroid,), nuc_loc)[0]
                for num in range(1, 4):
                    neighbor_nuc = nuc_prop[np.argsort(dist)[num]]
                    neiid = neighbor_nuc.label
                    nei_curr = curr[curr_label.index(neiid)]
                    if neiid not in prev_label:
                        break
                    nei_prev = prev[prev_label.index(neiid)]
                    nei_curr_sig = nei_curr.mean_intensity * nei_curr.area
                    nei_prev_sig = nei_prev.mean_intensity * nei_prev.area
                    if judge_bad(nei_curr_sig, nei_prev_sig, THRESCHANGE):
                        label[rp.coords[:, 0], rp.coords[:, 1]] = neiid
                        break
                    else:
                        pass
    return label
