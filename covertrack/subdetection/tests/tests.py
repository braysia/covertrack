import unittest
import sys
import os
import numpy as np
from os.path import join, basename, dirname, exists
sys.path.append(dirname(dirname(__file__)))
sys.path.append(dirname(dirname(dirname(__file__))))
import shutil
from mock import Mock
import subdetection_operations as operations


class TestUnitSubdetect(unittest.TestCase):
    def setUp(self):
        self.img = np.ones((5, 5))
        self.img[2:-2, 2:-2] = 2
        self.label = np.zeros((5, 5), np.uint16)
        self.img[2:-2, 2:-2] = 1
        self.holder = Mock()

    def tearDown(self):
        pass

    def test_call_op(self):
        dir_ops = dir(operations)
        dir_ops = [i for i in dir_ops if not i.startswith('__')]
        for d in dir_ops:
            if hasattr(getattr(operations, d), '__module__'):
                module = getattr(getattr(operations, d), '__module__')
                if 'subdetection_operations' in module:
                    func = getattr(operations, d)
                    label = func(self.img.copy(), self.label.copy(), self.holder)
                    self.assertEqual(self.img.shape, label.shape)

if __name__ == "__main__":
    unittest.main()
