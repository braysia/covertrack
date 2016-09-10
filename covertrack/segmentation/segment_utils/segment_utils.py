import numpy as np
import SimpleITK as sitk
from skimage.feature import peak_local_max
from covertrack.utils.seg_utils import skilabel


def lap_local_max(img, sigma_list, MIN_DIST, REL_THRES):
    img = np.uint16(img)
    lapimages = []
    for sig in sigma_list:
        simg = sitk.GetImageFromArray(img)
        nimg = sitk.LaplacianRecursiveGaussian(image1=simg, sigma=sig)
        lapimages.append(-sitk.GetArrayFromImage(nimg))

    image_cube = np.dstack(lapimages)
    local_maxima = peak_local_max(image_cube, min_distance=MIN_DIST, threshold_rel=REL_THRES,
                                  exclude_border=False, indices=False)

    local_maxima = local_maxima.sum(axis=2)
    local_maxima = skilabel(local_maxima)
    return local_maxima
