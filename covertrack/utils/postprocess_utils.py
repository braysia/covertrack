import numpy as np

def compute_segments(data):
    '''return a list of connected abs_id
    e.g. segments = [[1, 4, 7], [2, 3], [5, 6]]
    '''
    label_ids = np.unique([i.cell_id for i in data])
    segments = []
    for li in label_ids:
        segments.append([i.abs_id for i in data if i.cell_id == li])
    return segments

def extract_division_info_label_id(storage):
    daughters_cell_ids = [i.label_id for i in storage if not np.isnan(i.parent_id)]
    parent_ids = [i.parent_id for i in storage if not np.isnan(i.parent_id)]
    div_frame = [i.frame for i in storage if not np.isnan(i.parent_id)]
    return div_frame, daughters_cell_ids, parent_ids
