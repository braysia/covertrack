import unittest
import sys
import os
from os.path import join, basename, dirname, exists
sys.path.append(dirname(dirname(__file__)))
sys.path.append(dirname(dirname(dirname(__file__))))
from mock import Mock
import numpy as np
from covertrack.utils.seg_utils import calc_neck_score_thres, labels2outlines
from skimage.measure import regionprops
from covertrack.utils.seg_utils import calc_neck_score_thres_filtered, calc_shortest_step_coords


class TestNeckScore(unittest.TestCase):
    def setUp(self):
        template = np.zeros((7, 7), np.uint16)
        template[1:-1, 1:-1] = 1
        template[3, 1] = 0
        template[3, 5] = 0
        self.template = template

    def test_calc_neck_score(self):
        rp = regionprops(labels2outlines(self.template))
        score, coords = calc_neck_score_thres(rp[0].coords, edgelen=1)
        self.assertTrue((score == np.array([270, 270])).all())

    def test_calc_shortest_step(self):
        coords = np.array([[0, 0], [0, 1], [0, 2], [1, 2], [2, 2], [2, 1], [2, 0], [1, 0]])
        minstep = calc_shortest_step_coords(coords, coords[2, :], coords[6, :])
        self.assertEqual(minstep, 4)

    def test_calc_neck_score_thres_filtered(self):
        template = self.template.copy()
        template[4, 5] = 0
        rp = regionprops(labels2outlines(template))
        score, coords = calc_neck_score_thres_filtered(rp[0].coords, edgelen=1, steplim=15)
        self.assertEqual(len(score), 0)
        score, coords = calc_neck_score_thres_filtered(rp[0].coords, edgelen=1, steplim=4)
        self.assertEqual(len(score), 2)

if __name__ == '__main__':
    unittest.main()
