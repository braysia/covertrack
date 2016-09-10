import numpy as np
from munkres import munkres


def calc_cell_distance(cell1, cell2):
    return abs(cell1.prop.corr_x - cell2.prop.corr_x) + abs(cell1.prop.corr_y - cell2.prop.corr_y)


def calc_cell_massdiff(cell1, cell2):
    return (cell1.prop.total_intensity - cell2.prop.total_intensity)/cell2.prop.total_intensity


def find_one_to_one_assign(cost):
    (_, col1) = np.where([np.sum(cost, 0) == 1])
    cost[np.sum(cost, 1) != 1] = False
    (row, col2) = np.where(cost)
    good_curr_idx = [ci for ri, ci in zip(row, col2) if ci in col1]
    good_prev_idx = [ri for ri, ci in zip(row, col2) if ci in good_curr_idx]
    return good_curr_idx, good_prev_idx


def prepare_costmat(cost, costDie, costBorn):
    '''d is cost matrix,
    often distance matrix where rows are previous and columns as current
    d contains NaN where tracking of those two objects are not possible.
    costDie and costBorn'''
    cost[np.isnan(cost)] = np.Inf  # give a large cost.
    costDieMat = np.float64(np.diag([costDie]*cost.shape[0]))  # diagonal
    costBornMat = np.float64(np.diag([costBorn]*cost.shape[1]))
    costDieMat[costDieMat == 0] = np.Inf
    costBornMat[costBornMat == 0] = np.Inf

    costMat = np.ones((sum(cost.shape), sum(cost.shape)))*np.Inf
    costMat[0:cost.shape[0], 0:cost.shape[1]] = cost
    costMat[-cost.shape[1]:, 0:cost.shape[1]] = costBornMat
    costMat[0:cost.shape[0], -cost.shape[0]:] = costDieMat
    lowerRightBlock = cost.transpose()
    costMat[cost.shape[0]:, cost.shape[1]:] = lowerRightBlock
    return costMat


def call_lap(cost, costDie, costBorn):
    costMat = prepare_costmat(cost, costDie, costBorn)
    t = munkres(costMat)
    topleft = t[0:cost.shape[0], 0:cost.shape[1]]
    return topleft


def convert_coords_to_linear(x, shape):
    return np.ravel_multi_index(x, shape, order='C')


def flatlist(lis):
    return [i for j in lis for i in j]


def find_cells_overlap_linear_coords(cells, coords, shape):
    cells_overlapped = []
    for num, cell in enumerate(cells):
        lin_old_coords = [convert_coords_to_linear(i, shape) for i in cell.prop.coords]
        if [i for i in lin_old_coords if i in coords]:
            cells_overlapped.append(cell)
    return cells_overlapped


def find_coords_overlap_coords(cells, coords, shape):
    cells_overlap = find_cells_overlap_linear_coords(cells, coords, shape)
    lin_coords = []
    for cell in cells_overlap:
        lin_coords.append([convert_coords_to_linear(i, shape) for i in cell.prop.coords])
    return cells_overlap, flatlist(lin_coords)


def pick_closer_binarycostmat(binarymat, distmat):
    '''
    pick closer cells if there are two similar nucleus within area
    '''
    twonuc = np.where(np.sum(binarymat, 1) == 2)[0]
    for ti in twonuc:
        di = distmat[ti, :]
        bi = binarymat[ti, :]
        binarymat[ti, :] = min(di[bi]) == di
    return binarymat
