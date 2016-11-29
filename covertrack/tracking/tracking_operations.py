import numpy as np
from track_utils.cell_container import _distance_diff, _totalintensity_difference
from track_utils.cell_calculation import calc_cell_distance, calc_cell_massdiff
try:
    from covertrack.cell import CellListMaker
    from covertrack.utils.imreg import translation
    from covertrack.utils.seg_utils import skilabel, watershed
    from covertrack.segmentation.segment_utils.filters import sizefilterandopen
    from covertrack.utils.seg_utils import calc_neck_score_thres_filtered, labels2outlines, cut_neck
except:
    from cell import CellListMaker
    from utils.imreg import translation
    from utils.seg_utils import skilabel, watershed
    from segmentation.segment_utils.filters import sizefilterandopen
    from utils.seg_utils import calc_neck_score_thres_filtered, labels2outlines, cut_neck
from track_utils.cell_calculation import find_one_to_one_assign, call_lap, convert_coords_to_linear
from track_utils.cell_calculation import find_cells_overlap_linear_coords, flatlist, find_coords_overlap_coords
from track_utils.cell_calculation import pick_closer_binarycostmat
from skimage.segmentation import clear_border
from skimage.measure import regionprops
from logging import getLogger


'''
To track cells, simply set "cell.next = cell_in_curr_frame"

'''

logger = getLogger('covertrack.tracking')


def cell_nearest_neighbor(img, label, container, holder, DISPLACEMENT=100, MASSTHRES=0.2):
    """lazier implementation but may be easier and cleaner to write.
    A simple example for tracking.

    Args:
        DISPLACEMENT (int): The maximum distance (in pixel)
        MASSTHRES (float): The maximum difference of total intensity changes.
                           0.2 means it allows for 20% total intensity changes.
    """
    untr_prev, untr_curr = container.unlinked
    for cell in untr_prev:
        for cand in untr_curr:
            within_dist = calc_cell_distance(cell, cand) <= DISPLACEMENT
            within_mass = abs(calc_cell_massdiff(cell, cand)) < MASSTHRES
            if within_dist and within_mass:
                cell.next = cand
    return container


def nearest_neighbor(img, label, container, holder, DISPLACEMENT=100, MASSTHRES=0.2):
    """Link two cells if they are the only two cells within DISPLACEMENT and MASSTHRES. 

    Args:
    DISPLACEMENT (int): The maximum distance (in pixel)
    MASSTHRES (float): The maximum difference of total intensity changes.
                        0.2 means it allows for 20% total intensity changes.
    """

    DISPLACEMENT = DISPLACEMENT
    MASSTHRES = MASSTHRES
    withinarea = container._dist_unlinked < DISPLACEMENT
    withinmass = abs(container._masschange_unlinked) < MASSTHRES
    binarymat = withinarea * withinmass
    binarycost = binarymat
    good_curr_idx, good_prev_idx = find_one_to_one_assign(binarycost)
    prev_cells, curr_cells = container.unlinked
    for ci, pi in zip(good_curr_idx, good_prev_idx):
        prev_cells[pi].next = curr_cells[ci]
    return container


def run_lap(img, label, container, holder, DISPLACEMENT=100, MASSTHRES=0.2):
    '''Linear assignment problem for mammalian cells.
    Cost matrix is simply the distance.
    costDie and costBorn are variables changing over frame. Update it through holder.

    Args:
    DISPLACEMENT (int): The maximum distance (in pixel)
    MASSTHRES (float):  The maximum difference of total intensity changes.
                        0.2 means it allows for 20% total intensity changes.
    '''
    dist = container._dist_unlinked
    massdiff = container._masschange_unlinked
    '''search radius is now simply set by maximum displacement possible.
    In the future, I might add data-driven approach mentioned in LAP paper (supple pg.7)'''
    dist[dist > DISPLACEMENT] = np.Inf  # assign a large cost for unlikely a pair
    # dist[abs(massdiff) > MASSTHRES] = np.Inf
    cost = dist
    if cost.shape[0] == 0 or cost.shape[1] == 0:
        return container

    # Define initial costBorn and costDie in the first frame
    if not hasattr(holder, 'cost_born') or hasattr(holder, 'cost_die'):
        holder.cost_born = np.percentile(cost[~np.isinf(cost)], 80)
        holder.cost_die = np.percentile(cost[~np.isinf(cost)], 80)
    # try:
    binary_cost = call_lap(cost, holder.cost_die, holder.cost_born)
    # The first assignment of np.Inf is to reduce calculation of linear assignment.
    # This part will make sure that cells outside of these range do not get connected.
    binary_cost[abs(massdiff) > MASSTHRES] = False
    binary_cost[dist > DISPLACEMENT] = False

    gp, gc = np.where(binary_cost)
    good_prev_idx, good_curr_idx = list(gp), list(gc)
    prev_cells, curr_cells = container.unlinked
    for ci, pi in zip(good_curr_idx, good_prev_idx):
        prev_cells[pi].next = curr_cells[ci]

    # update cost
    dist = container._dist_unlinked
    if dist.size:
        cost = np.max(dist)*1.05
        if cost != 0:  # solver freezes if cost is 0
            holder.cost_born, holder.cost_die = cost, cost
    return container


def watershed_distance(img, label, container, holder, ERODI=5,
                       DEBRISAREA=50, DISPLACEMENT=50, MASSTHRES=0.2):
    '''
    Adaptive segmentation by using tracking informaiton.
    watershed existing label, meaning make a cut at the deflection.
    After the cuts, objects will be linked if they are within DISPLACEMENT and MASSTHRES.
    If two candidates are found, it will pick a closer one.

    Args:
    ERODI (int):        Erosion size element for generating watershed seeds.
                        Smaller ERODI will allow more cuts.
    DISPLACEMENT (int): The maximum distance (in pixel)
    MASSTHRES (float):  The maximum difference of total intensity changes.
                        0.2 means it allows for 20% total intensity changes.
    '''

    untr_prev, untr_curr = container.unlinked
    mask_untracked = container._label_untracked.astype(bool)
    wshed_label = watershed(mask_untracked, ERODI)
    wshed_label = skilabel(sizefilterandopen(wshed_label, DEBRISAREA, np.Inf, 0))
    newcells = CellListMaker(img, wshed_label, holder, holder.frame).make_list()
    distanceUntracked = _distance_diff(untr_prev, newcells)
    masschangeUntracked = _totalintensity_difference(untr_prev, newcells)

    withinarea = distanceUntracked < DISPLACEMENT
    withinmass = abs(masschangeUntracked) < MASSTHRES
    withinareaMass = withinarea * withinmass

    withinareaMass = pick_closer_binarycostmat(withinareaMass, distanceUntracked)
    good_curr_idx, good_prev_idx = find_one_to_one_assign(withinareaMass)

    # update the link
    for ci, pi in zip(good_curr_idx, good_prev_idx):
        untr_prev[pi].next = newcells[ci]
    # Get all linear coordinates from good newly segmented cells
    good_curr_coords = [newcells[n].prop.coords for n in good_curr_idx]
    lin_curr_coords = [convert_coords_to_linear(i, holder.img_shape) for i in flatlist(good_curr_coords)]
    # find cells in old mask (in current) that overlaps with good new cells
    old_cells_to_remove, lin_old_coords_remove = find_coords_overlap_coords(untr_curr, lin_curr_coords, holder.img_shape)
    # find cells in new mask which overlaps with the cells in old mask
    newcells_to_update = find_cells_overlap_linear_coords(newcells, lin_old_coords_remove, holder.img_shape)
    # remove old cells
    for old_cell in old_cells_to_remove:
        container.curr_cells.remove(old_cell)
    # add new cells
    container.curr_cells.extend(newcells_to_update)
    return container


def jitter_correction_label(img, label, container, holder):
    '''Simple but simpler jitter correction based on markers.
    It would not work if you have too few objects. This will add jitters to corr_x and corr_y.
    Values of jitter is relative to the first frame, so they accumulate jitters
    in consecutive frames in holder.jitter.
    Add this as a first algorithm in track_args when use.
    '''
    if not hasattr(holder, 'prev_label'):
        holder.prev_label = label
    if np.any(holder.prev_label):
        prevlabel = holder.prev_label
        ocCurr = label
        ocPrev = prevlabel
        jitter = translation(ocPrev, ocCurr)
        if not hasattr(holder, 'jitter'):
            holder.jitter = [0, 0]
        jitter = translation(ocPrev, ocCurr)
        for i in (0, 1):
            holder.jitter[i] = holder.jitter[i] + jitter[i]
        for cell in container.curr_cells:
            cell.prop.jitter_x = holder.jitter[1]
            cell.prop.jitter_y = holder.jitter[0]
            cell.prop.corr_x = cell.prop.corr_x + holder.jitter[1]
            cell.prop.corr_y = cell.prop.corr_y + holder.jitter[0]
    logger.debug("\t\tA field moved {0} pix to x and {1} pix to y".format(*holder.jitter))
    return container


def jitter_correction_label_at_frame(img, label, container, holder, FRAME=1):
    """
        FRAME (List(int) or int): a list of frames to run jitter correction
    """
    if isinstance(FRAME, int):
        FRAME = [FRAME, ]
    if not hasattr(holder, 'prev_label'):
        holder.prev_label = label
    if holder.frame in FRAME:
        container = jitter_correction_label(img, label, container, holder)
    logger.debug("\t\tA field moved {0} pix to x and {1} pix to y".format(*holder.jitter))
    return container


def track_neck_cut(img, label, container, holder, ERODI=5, DEBRISAREA=50, DISPLACEMENT=50,
                   MASSTHRES=0.2, LIM=10, EDGELEN=5, THRES_ANGLE=180, STEPLIM=10):
        """
        Adaptive segmentation by using tracking informaiton.
        Separate two objects by making a cut at the deflection. For each points on the outline,
        it will make a triangle separated by EDGELEN and calculates the angle facing inside of concave.

        EDGELEN (int):      A length of edges of triangle on the nuclear perimeter.
        THRES_ANGLE (int):  Define the neck points if a triangle has more than this angle.
        STEPLIM (int):      points of neck needs to be separated by at least STEPLIM in parimeters.
        """

        untr_prev, untr_curr = container.unlinked
        label_untracked = container._label_untracked
        unique_labels = np.unique(label_untracked)
        unique_labels = unique_labels[unique_labels > 0]
        newcells = []
        all_new_cells = []
        for label_id in unique_labels:
            mask = label_untracked == label_id
            cl_label = clear_border(mask)
            outlines = labels2outlines(cl_label).astype(np.uint16)
            rps = regionprops(outlines)
            rps = [i for i in rps if i.perimeter > STEPLIM]
            for cell in rps:
                score, coords = calc_neck_score_thres_filtered(cell.coords, edgelen=EDGELEN, thres=THRES_ANGLE, steplim=STEPLIM)
                if len(score) > 1:
                    r0, c0 = coords[0, :]
                    if coords.shape[0] > LIM:
                        coords = coords[:LIM, :]
                    for cand in coords[1:, :]:
                        untr_prev = container.unlinked[0]

                        cut_label = skilabel(cut_neck(cl_label, r0, c0, cand[0], cand[1]), conn=1)
                        new_cells_temp = CellListMaker(img, cut_label, holder, holder.frame).make_list()
                        if len(new_cells_temp) > 1:
                            distanceUntracked = _distance_diff(untr_prev, new_cells_temp)
                            masschangeUntracked = _totalintensity_difference(untr_prev, new_cells_temp)

                            withinarea = distanceUntracked < DISPLACEMENT
                            withinmass = abs(masschangeUntracked) < MASSTHRES
                            withinareaMass = withinarea * withinmass

                            withinareaMass = pick_closer_binarycostmat(withinareaMass, distanceUntracked)
                            good_curr_idx, good_prev_idx = find_one_to_one_assign(withinareaMass)
                            if len(good_curr_idx) > 0:
                                # update the link
                                all_new_cells.append(new_cells_temp)
                                for ci, pi in zip(good_curr_idx, good_prev_idx):
                                    newcells.append(new_cells_temp[ci])
                                    untr_prev[pi].next = new_cells_temp[ci]
                                break


        good_curr_coords = [n.prop.coords for n in newcells]
        lin_curr_coords = [convert_coords_to_linear(i, holder.img_shape) for i in flatlist(good_curr_coords)]
        # find cells in old mask (in current) that overlaps with good new cells
        old_cells_to_remove, lin_old_coords_remove = find_coords_overlap_coords(untr_curr, lin_curr_coords, holder.img_shape)
        # find cells in new mask which overlaps with the cells in old mask
        all_new_cells = [i for j in all_new_cells for i in j]
        newcells_to_update = find_cells_overlap_linear_coords(all_new_cells, lin_old_coords_remove, holder.img_shape)
        # remove old cells
        for old_cell in old_cells_to_remove:
            container.curr_cells.remove(old_cell)
        # add new cells
        container.curr_cells.extend(newcells_to_update)
        return container


def back_track(img, label, container, holder, BACKFRAME=None):
    """
    Implement tracking from the BACKFRAME frame to the beginning and then beginning to the end.
    By running this, it will find a better segmentation in the first frame if you combine
    with the adaptive segmentation such as track_neck_cut or watershed_distance.
    This modifies self.pathset in call_tracking and dynamically change the behavior the of the loop.
    If you have 4 frames, the overall loop changes from [0, 1, 2, 3] to [0, 1, 3, 2, 1, 0, 1, 2, 3].

    Args:
        BACKFRAME (int): decide which frames to start the back-tracking.
    """
    ind = holder.pathset.index(holder.imgpath)
    if not hasattr(holder, 'back_flag'):
        holder.back_flag = True
    if holder.back_flag:
        for i in holder.pathset:
            if holder.back_flag:
                for ii in holder.pathset[:BACKFRAME]:
                    holder.pathset.insert(ind, ii)
                    holder.back_flag = False
                holder.pathset.insert(ind, holder.pathset[-1])
    return container