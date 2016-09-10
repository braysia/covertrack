import numpy as np


def find_one_to_one_assign(cost):
    (_, col1) = np.where([np.sum(cost, 0) == 1])
    cost[np.sum(cost, 1) != 1] = False
    (row, col2) = np.where(cost)
    goodCurrent = [ci for ri, ci in zip(row, col2) if ci in col1]
    goodPrevious = [ri for ri, ci in zip(row, col2) if ci in goodCurrent]
    return goodCurrent, goodPrevious
