import unittest
import numpy as np
from mock import Mock
from os.path import dirname
import sys
sys.path.append(dirname(dirname(__file__)))
sys.path.append(dirname(dirname(dirname(__file__))))
from covertrack.cell import Cell
from posttrack_utils.traces import construct_traces_based_on_next, convert_traces_to_storage
from posttrack_utils.traces import assign_next_and_abs_id_to_storage, connect_parent_daughters


class CellMock(Cell):
    '''Cell object without prop'''
    def __init__(self, frame):
        self.frame = frame
        self.cell_id = None
        self.parent = None
        self._next = None
        self.previous = None
        self.prop = Mock()


class Test_posttrack_utils_traces(unittest.TestCase):
    def setUp(self):
        storage = []
        for i in range(10):
            storage.append(CellMock(i))
        for i in [1, 3, 5, 7]:
            storage[i].next = storage[i+2]
        for i in [2, 4, 6]:
            storage[i].next = storage[i+2]
        self.storage = storage
        '''This supposes to make traces like
        [[cell1, cell3, cell5, cell7, cell9],
        [cell2, cell4, cell6, cell8]
        [cell0]]
        '''

    def test_construct_traces_based_on_next(self):
        traces = construct_traces_based_on_next(self.storage)
        self.assertTrue(isinstance(traces, list))
        self.assertEqual(len(traces), 3)
        traces.sort(key=len)
        self.assertEqual(len(traces[0]), 1)
        self.assertEqual(len(traces[1]), 4)
        self.assertEqual(len(traces[2]), 5)

    def test_convert_traces_to_storage(self):
        traces = construct_traces_based_on_next(self.storage)
        new_storage = convert_traces_to_storage(traces)
        traces = construct_traces_based_on_next(new_storage)
        self.assertTrue(isinstance(traces, list))
        self.assertEqual(len(traces), 3)
        traces.sort(key=len)
        self.assertEqual(len(traces[0]), 1)
        self.assertEqual(len(traces[1]), 4)
        self.assertEqual(len(traces[2]), 5)


class Test_posttrack_utils_storage(unittest.TestCase):
    def setUp(self):
        storage = []
        storage.append([CellMock(1), CellMock(1)])
        storage.append([CellMock(2), CellMock(2), CellMock(2)])
        storage.append([CellMock(3)])
        storage[0][0].prop.label_id = 1
        storage[1][1].prop.label_id = 1
        storage[2][0].prop.label_id = 1
        storage[0][1].prop.label_id = 2
        storage[1][0].prop.label_id = 2
        storage[1][2].prop.label_id = 3
        self.storage = storage

    def test_assign_next_and_abs_id_to_storage(self):
        storage = assign_next_and_abs_id_to_storage(self.storage)
        self.assertEqual(storage[0].next.next.frame, 3)
        self.assertEqual(set(range(1, 7)), set([i.abs_id for i in storage]))


class Test_posttrack_utils_parents(unittest.TestCase):
    def setUp(self):
        storage = []
        storage.append([CellMock(1), CellMock(1)])
        storage.append([CellMock(2), CellMock(2)])

        # Assign unique label_id to all cells
        lab = 1
        for i in storage:
            for ii in i:
                ii.prop.label_id = lab
                lab += 1

        storage[1][0].parent = storage[0][0]
        storage[1][1].parent = storage[0][0]
        self.storage = storage

    def test_connect_parent_daughters(self):
        traces = construct_traces_based_on_next([i for j in self.storage for i in j])
        self.assertEqual(len(traces), 4)
        label_ids = [i.prop.label_id for j in traces for i in j]
        self.assertNotEqual(label_ids.count(1), 2)
        traces = connect_parent_daughters(traces)
        self.assertEqual(len(traces), 3)
        label_ids = [i.prop.label_id for j in traces for i in j]
        self.assertEqual(label_ids.count(1), 2)
