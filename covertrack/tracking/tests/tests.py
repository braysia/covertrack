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

class TestUnitTrack(unittest.TestCase):
    def setUp(self):
        self.img = np.ones((5, 5))
        self.img[2:-2, 2:-2] = 2
        self.label = np.zeros((5, 5), np.uint16)
        self.label[1, 1] = 1
        self.label[2, 2] = 2
        self.label[3, 3] = 3
        self.holder = Holder()
        self.holder.img_shape = self.img.shape
        self.holder.prev_img = self.img
        self.holder.prev_label = self.label
        self.holder.frame = 1

    def tearDown(self):
        pass

    def test_call_op(self):
        dir_ops = dir(operations)
        dir_ops = [i for i in dir_ops if not i.startswith('__')]
        for d in dir_ops:
            if hasattr(getattr(operations, d), '__module__'):
                module = getattr(getattr(operations, d), '__module__')
                if 'tracking_operations' in module:
                    container = Container(self.img.shape)
                    container.curr_cells = CellListMaker(self.img, self.label, 1).make_list()
                    container.prev_cells = CellListMaker(self.img, self.label, 0).make_list()

                    func = getattr(operations, d)
                    con = func(self.img.copy(), self.label.copy(), container, self.holder)
                    self.assertTrue(isinstance(con, Container))

if __name__ == "__main__":
    unittest.main()
