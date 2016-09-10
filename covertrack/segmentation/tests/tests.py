import unittest
import sys
import os
import numpy as np
from os.path import join, basename, dirname, exists
sys.path.append(dirname(dirname(__file__)))
sys.path.append(dirname(dirname(dirname(__file__))))
import shutil
from mock import Mock
import segmentation_operations


class TestUnitSegment(unittest.TestCase):
    def setUp(self):
        self.img = np.ones((5, 5))
        self.img[2:-2, 2:-2] = 2

    def tearDown(self):
        pass

    def test_call_op(self):
        dir_ops = dir(segmentation_operations)
        dir_ops = [i for i in dir_ops if not i.startswith('__')]
        for d in dir_ops:
            if hasattr(getattr(segmentation_operations, d), '__module__'):
                module = getattr(getattr(segmentation_operations, d), '__module__')
                if 'segmentation_operations' in module:
                    func = getattr(segmentation_operations, d)
                    label = func(self.img.copy(), Mock())
                    self.assertEqual(self.img.shape, label.shape)

if __name__ == "__main__":
    unittest.main()
