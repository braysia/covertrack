import numpy as np
from operator import attrgetter
import copy
from itertools import izip
import sys

sys.setrecursionlimit(10000)


class TracesController(object):
    def __init__(self, traces):
        self.traces = traces
        frames = [cell.frame for cell in [i[-1] for i in traces]]
        self.last_frame = max(frames)

    def disappeared(self):
        cells = [i[-1] for i in self.traces]
        return [cell for cell in cells if cell.frame < self.last_frame]

    def appeared(self):
        cells = [i[0] for i in self.traces]
        return [cell for cell in cells if cell.frame > 0]

    def pairwise_dist(self):
        dis_x, dis_y = retrieve_xy(self.disappeared())
        app_x, app_y = retrieve_xy(self.appeared())
        return calc_dist(app_x, app_y, dis_x, dis_y)

    def pairwise_frame(self):
        '''disappeared in row, appeared in col'''
        return calc_diffframe(self.appeared(), self.disappeared())

    def pairwise_mass(self):
        return calc_massdiff(self.appeared(), self.disappeared())


def calc_dist(xObj1, yObj1, xObj2, yObj2):
    xObj2Tile = np.tile(xObj2, (len(xObj1), 1)).astype(np.float32)
    yObj2Tile = np.tile(yObj2, (len(yObj1), 1)).astype(np.float32)
    dist = abs(xObj2Tile.T - xObj1) + abs(yObj2Tile.T - yObj1)
    return dist


def calc_diffframe(cells1, cells2):
    frameObj1 = np.array([i.frame for i in cells1])
    frameObj2 = np.array([i.frame for i in cells2])
    frameObj2Tile = np.tile(frameObj2, (len(frameObj1), 1)).T
    framediff = frameObj1 - frameObj2Tile
    return framediff


def calc_massdiff(obj1, obj2):
    massObj1 = np.array([i.prop.total_intensity for i in obj1])
    massObj2 = np.array([i.prop.total_intensity for i in obj2])
    massObj2Tile = np.tile(massObj2, (len(massObj1), 1)).T
    massdiff = (massObj1 - massObj2Tile) / massObj2Tile
    return massdiff


def retrieve_xy(cells):
    x = np.array([i.prop.corr_x for i in cells])
    y = np.array([i.prop.corr_y for i in cells])
    return x, y

# def construct_traces_based_on_label(storage):
#     unique_label_ids = np.unique([i.prop.label_id for i in storage])
#     traces = []
#     for li in unique_label_ids:
#         traces.append(Trace([cell for cell in storage if cell.prop.label_id == li]))
#     return traces


def assign_next_and_abs_id_to_storage(storage):
    '''Set next if two cells in consecutive frames have the same label_id'''
    for prev_cells, curr_cells in zip(storage[0:-1], storage[1:]):
        for prev_cell in prev_cells:
            for curr_cell in curr_cells:
                if prev_cell.prop.label_id == curr_cell.prop.label_id:
                    prev_cell.next = curr_cell
    storage = [i for j in storage for i in j]
    # set abs_id from 1 to max
    [setattr(si, 'abs_id', i+1) for i, si in enumerate(storage)]
    return storage


def construct_traces_based_on_next(storage):
    '''
    Convert storage to traces.
    traces is a list of lists containing cells with same label_id.
    If cell1.next = cell2 and cell2.next = cell3, cell4.next = cell5,
    then traces = [[cell1, cell2, cell3], [cell4, cell5]].
    storage has to be sorted based on frame.
    '''
    traces = []
    for cell in storage:
        cells = [cell]
        while cell.next is not None:
            cell = cell.next
            cells.append(storage.pop(storage.index(cell)))
        traces.append(cells)
    return traces


def convert_traces_to_storage(traces):
    '''Convert traces to storage.
    '''
    storage = [i for j in traces for i in j]
    return sorted(storage, key=attrgetter('frame'))


def connect_parent_daughters(traces):
    '''Make sure to use this after saving labels.
    Parents will be duplicated, and Cell_id will be updated
    Assume traces = [[cell2, cell3], [cell4, cell5]], [cell6, cell7]]
    and cell4.parent = cell3, cell6.parent = cell3.
    Then this converts traces to
    [[cell2, cell3, cell4, cell5], [cell2, cell3, cell6, cell7]].
    '''
    # traces = construct_traces_based_on_next(storage)
    # Concatenate parent and daughters
    parent_cells = [i[0].parent for i in traces if i[0].parent is not None]
    traces_without_parent = []
    parental_traces = []
    while traces:
        trace = traces.pop(0)
        if trace[-1] in parent_cells:
            parental_traces.append(trace)
        else:
            traces_without_parent.append(trace)
    for trace in parental_traces:
        trace2 = [copy.copy(i) for i in trace]
        parent_traces = [trace, trace2]
        daughter_traces = [i for i in traces_without_parent if i[0].parent in trace]
        for parent_trace, daughter_trace in izip(parent_traces, daughter_traces):
            daughter_trace.extend(parent_trace)
            daughter_trace.sort(key=attrgetter('frame'))
    return traces_without_parent


def extract_division_info_label_id(storage):
    daughter_cells = [i for i in storage if i.parent is not None]
    parent_ids = [i.parent_id for i in storage if not np.isnan(i.parent_id)]
    div_frame = [i.frame for i in storage if not np.isnan(i.parent_id)]
    return div_frame, daughters_cell_ids, parent_ids


def division_frames_and_cell_ids(storage):
    daughter_cells = [i for i in storage if i.parent is not None]
    divided_cell_ids = [i.cell_id for i in daughter_cells]
    div_frame = [i.frame for i in daughter_cells]
    return div_frame, divided_cell_ids


def retrieve_dist(obj1, obj2):
    xObj1, yObj1 = retrieve_coor(obj1)
    xObj2, yObj2 = retrieve_coor(obj2)
    dist = calc_dist(xObj1, yObj1, xObj2, yObj2)
    return dist

def retrieve_coor(obj):
    xObj = np.array([i.corr_x for i in obj]).astype(np.float32)
    yObj = np.array([i.corr_y for i in obj]).astype(np.float32)
    return xObj, yObj

def calc_dist(xObj1, yObj1, xObj2, yObj2):
    xObj2Tile = np.tile(xObj2, (len(xObj1), 1)).astype(np.float32)
    yObj2Tile = np.tile(yObj2, (len(yObj1), 1)).astype(np.float32)
    dist = abs(xObj2Tile.T - xObj1) + abs(yObj2Tile.T - yObj1)
    return dist

def calc_framediff(obj1, obj2):
    frameObj1 = np.array([i.frame for i in obj1])
    frameObj2 = np.array([i.frame for i in obj2])
    frameObj2Tile = np.tile(frameObj2, (len(frameObj1), 1)).T
    framediff = frameObj1 - frameObj2Tile
    return framediff

def calc_massdiff(obj1, obj2):
    massObj1 = np.array([i.prop.total_intensity for i in obj1])
    massObj2 = np.array([i.prop.total_intensity for i in obj2])
    massObj2Tile = np.tile(massObj2, (len(massObj1), 1)).T
    massdiff = (massObj1 - massObj2Tile) / massObj2Tile
    return massdiff
