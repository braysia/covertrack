import unittest
import sys
import os
import numpy as np
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
sys.path.append(dirname(dirname(dirname(abspath(__file__)))))
import shutil
from mock import Mock
import tracking_operations as operations
from track_utils.cell_container import Container
from covertrack.cell import CellListMaker
from call_tracking import Holder
# from track_utils.cell_calculation import
from munkres import munkres
from track_utils.cell_calculation import call_lap


np.random.seed(1)

class TestTrackOps(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    # def test_munkres(self):
    #     for i in range(3):
    #         cost = np.random.random((5, 5))
    #         mask = np.random.random((5, 5)) < 0.5
    #         cost[mask] = np.Inf
    #         lin_arr = munkres(cost)
    #         self.assertFalse(lin_arr[mask].any())

    def test_call_lap(self):
        for i in range(100):
            cost = np.random.random((5, 5))
            mask = np.random.random((5, 5)) < 0.5
            cost[mask] = np.Inf
            binary_cost = call_lap(cost, 0.6, 0.6)
            idx, idy = np.where(binary_cost)
            template = np.zeros(cost.shape, np.bool)
            template[idx, idy] = True
            self.assertFalse(template[mask].any())




if __name__ == "__main__":
    unittest.main()
