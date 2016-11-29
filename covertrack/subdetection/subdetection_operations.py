import numpy as np
import sys
from os.path import dirname, abspath, basename, join
sys.path.append(dirname(abspath(__file__)))
from subdetect_utils.subdect_utils import dilate_to_cytoring, dilate_to_cytoring_buffer
from subdetect_utils.subdect_utils import double_propagate
from scipy.ndimage import imread
from subdetect_utils.subdect_utils import gradient_anisotropic, homogenize_cell_intensity_N4, propagate_and_cleanup
try:
    from covertrack.utils.seg_utils import adaptive_thresh
except:
    from utils.seg_utils import adaptive_thresh
from subdetect_utils.subdect_utils import label_nearest, label_high_pass
from subdetect_utils.subdect_utils import repair_sal
from skimage.morphology import binary_dilation, binary_closing
from scipy.ndimage.filters import minimum_filter


def ring_dilation(img, label, holder, MARGIN=0, RINGWIDTH=4):
    """Create a ring around label.
    :param RINGWIDTH: Width of rings
    :param MARGIN: A region of rings is ignored if they are within MARGIN pixels away from label.

    Examples:
        >>> arr = np.zeros((5, 5));arr[2, 2] = 10
        >>> ring_dilation(None, arr, None, MARGIN=1, RINGWIDTH=2)
        array([[ 0, 10, 10, 10,  0],
               [10,  0,  0,  0, 10],
               [10,  0,  0,  0, 10],
               [10,  0,  0,  0, 10],
               [ 0, 10, 10, 10,  0]], dtype=uint16)
    """
    return dilate_to_cytoring(label, RINGWIDTH, MARGIN)


def ring_dilation_buffer(img, label, holder, MARGIN=0, RINGWIDTH=4, BUFFER=2):
    return dilate_to_cytoring_buffer(label, RINGWIDTH, MARGIN, BUFFER)


def ring_dilation_above_thres(img, label, holder, MARGIN=2, RINGWIDTH=4,
                              EXTRA_RINGWIDTH=15, THRES=50):
    sub_label = dilate_to_cytoring(label, RINGWIDTH, MARGIN)
    extra_sub_label = dilate_to_cytoring(label, EXTRA_RINGWIDTH, RINGWIDTH)
    extra_sub_label[img < THRES] = 0
    return sub_label + extra_sub_label


def ring_dilation_above_offset_buffer(img, label, holder, MARGIN=0, RINGWIDTH=4, BUFFER=2,
                                      OFFSET=100, FILSIZE=50):
    """Dilate from label to make a ring.
    Calculate the local minimum as a background, and if image is less brighter
    than background + offset, remove the region from the ring.
    """
    sub_label = dilate_to_cytoring_buffer(label, RINGWIDTH, MARGIN, BUFFER)
    minimg = minimum_filter(img, size=FILSIZE)
    sub_label[img < (minimg + OFFSET)] = 0
    return sub_label

def ring_dilation_above_thres_buffer_extra(img, label, holder, MARGIN=2, RINGWIDTH=4, EXTRA_RINGWIDTH=7, THRES=np.Inf, BUFFER=5):
    sub_label = dilate_to_cytoring_buffer(label, RINGWIDTH, MARGIN, BUFFER)
    extra_sub_label = dilate_to_cytoring_buffer(label, EXTRA_RINGWIDTH, RINGWIDTH, BUFFER)
    extra_sub_label[img < THRES] = 0
    sub_label = sub_label + extra_sub_label
    return sub_label

def ring_dilation_above_adaptive(img, label, holder, MARGIN=0, RINGWIDTH=4, BUFFER=2, RATIO=1.05, FILSIZE=50):
    sub_label = dilate_to_cytoring_buffer(label, RINGWIDTH, MARGIN, BUFFER)
    bw = adaptive_thresh(img, RATIO=RATIO, FILTERINGSIZE=FILSIZE)
    sub_label[-bw] = 0
    return sub_label



def cytoplasm_double_propagate(img, label, holder, FIR_THRES=1000, SEC_THRES=500, OPEN=40):
    """"segment cytoplasm by using propagate twice.
    FIR_THRES for a main compartment of the cells (bright part).
    SEC_THRES for a secondary compartment which are darker, something barely above background.
    If expression level is very different from cells to cells, then you may need to do
    correct illumination before thresholding... Probably applying OPEN at the beginning.
    """
    sub_label = double_propagate(img, label, FIR_THRES, SEC_THRES, OPEN)
    # You can add some cleaning or dilation i
    sub_label[label > 0] = 0
    return sub_label


# def cytoplasm_propagagate(img, label, holder):
#     imgc = img.copy()
#     if len(imgc.shape) == 2:
#         imgc = np.expand_dims(imgc, axis=2)
#     store = []
#     for dim in range(imgc.shape[2]):
#         img = imgc[:, :, dim]
#         # Dirty
#         all_label = imread(join(holder.argdict['outputdir'], 'segmented', basename(holder.seed_imgpath)))
#         filimg = gradient_anisotropic(img)
#         bw = adaptive_thresh(filimg, RATIO=1.05, FILTERINGSIZE=50)
#         filimg = homogenize_cell_intensity_N4(filimg, bw)
#         bw = adaptive_thresh(filimg, RATIO=1.05, FILTERINGSIZE=50)
#         store.append(propagate_and_cleanup(filimg, label, all_label, bw))
#     sub_label = np.max(np.dstack(store), axis=2).astype(np.uint16)
#     return sub_label


def segment_bacteria(img, nuc, holder, slen=3, SIGMA=0.5, THRES=50, CLOSE=3, THRESCHANGE=1000):
    label = label_high_pass(img, slen=slen, SIGMA=SIGMA, THRES=50, CLOSE=3)
    if label.any():
        label, comb, nuc_prop, nuc_loc = label_nearest(img, label, nuc)
        if not hasattr(holder, 'plabel'):
            holder.plabel = label
            holder.pcomb = comb
            holder.pimg = img
            return label.astype(np.uint16)
        # repair
        label = repair_sal(img, holder.pimg, comb, holder.pcomb, label, nuc_prop, nuc_loc, THRESCHANGE)
    return label.astype(np.uint16)
