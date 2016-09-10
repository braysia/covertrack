import unittest
import sys
import os
from os.path import join, basename, dirname, exists
sys.path.append(dirname(dirname(__file__)))
sys.path.append(dirname(dirname(dirname(__file__))))
from mock import Mock
import numpy as np
from pairwise import one_to_one_assignment, one_to_two_assignment


class TestAssignment(object):
    def setUp(self):
        self.arr = np.zeros((4, 4), bool)
        self.arr[0, 1:3] = True
        self.arr[2, 2:3] = True
        self.arr[3, 0] = True
        self.cost = np.arange(0, 16).reshape(self.arr.shape)
        self.cost[0, 2] = 1

    def test_one_to_one(self):
        one_to_one_assignment(self.arr, self.cost)

class TestAssignmentOneToTwo(object):
    def setUp(self):
        self.arr = np.zeros((4, 4), bool)
        self.arr[0, 1:3] = True
        self.arr[2, 2:3] = True
        self.arr[3, 0] = True
        self.cost = np.arange(0, 16).reshape(self.arr.shape)
        self.cost[0, 2] = 1

    def test_one_to_two(self):
        one_to_two_assignment(self.arr, self.cost)


if __name__ == '__main__':
    unittest.main()
