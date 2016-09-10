import unittest
import numpy as np
from mock import Mock
from os.path import dirname
import sys
sys.path.append(dirname(dirname(__file__)))
from covertrack.cell import CellListMaker, CellListMakerScalar, Cell, Prop


class Test_CellListMaker(unittest.TestCase):
    def setUp(self):
        self.img = np.ones((10, 10))
        self.label = np.zeros((10, 10), np.int16)
        self.label[2:4, 3:5] = 1
        self.label[2:4, 8:10] = 2
        self.param = Mock()
        self.frame = 0

    def test_cell_list_CellListMaker(self):
        clm = CellListMaker(self.img, self.label, self.param, self.frame)
        celllist = clm.make_list()
        self.assertTrue(isinstance(celllist, list))
        self.assertTrue(isinstance(celllist[0], Cell))

    # def test_no_label_CellListMaker(self):
    #     zlabel = np.zeros((10, 10), np.int16)
    #     clm = CellListMaker(self.img, zlabel, self.param, self.frame)
    #     with self.assertRaises(Exception):
    #         clm.make_list()

    def test_isscalar_CellListMakerScalar(self):
        clm = CellListMakerScalar(self.img, self.label, self.param, self.frame)
        celllist = clm.make_list()
        cell = celllist[0]
        props = [i for i in dir(cell.prop) if not i.startswith('_')]
        for i in props:
            self.assertTrue(np.isscalar(getattr(cell.prop, i)))


class Cell_without_Prop(Cell):
    '''Mock class for Test_Cell_Next_Previous'''
    def __init__(self, frame):
        self.frame = frame
        self.cell_id = None
        self.parent = None
        self._next = None
        self.previous = None


class Test_Cell_Next_Previous(unittest.TestCase):
    def test_cell_next_previous(self):
        cell1 = Cell_without_Prop(0)
        cell2 = Cell_without_Prop(1)
        cell3 = Cell_without_Prop(2)
        cell1.next = cell2
        cell2.next = cell3
        self.assertEqual(cell1.next, cell2)
        self.assertEqual(cell1.next.next, cell3)
        self.assertEqual(cell3.previous, cell2)




