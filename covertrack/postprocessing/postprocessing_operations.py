from abc import ABCMeta, abstractmethod
import numpy as np
from covertrack.utils.postprocess_utils import compute_segments
from itertools import combinations
from covertrack.utils.tracking_utils import find_one_to_one_assign
from covertrack.utils.pairwise import one_to_one_assignment, one_to_two_assignment
from posttrack_utils.traces import TracesController
from posttrack_utils.traces import construct_traces_based_on_next, convert_traces_to_storage
from operator import attrgetter


def cut_short_traces(traces, holder, minframe=4):
    '''
    To count the traces including divison, after constructing traces using next,
    stick traces by using parent (need to deepcopy the parents temporally...).
    If 10 images are available and want to filter any partial traces, set minframe=9.
    '''
    if holder.num_frame <= minframe:
        # FIXME: change this to logging?
        print "too few frames available to cut short traces"
        return traces

    ''' Calculate the largest frame differences so it will go well with gap closing'''
    trhandler = TracesController(traces)
    storage = convert_traces_to_storage(trhandler.traces)
    temp_traces = construct_traces_based_on_next(storage[:])

    # Concatenate parent and daughters
    parent_cell = [i[0].parent for i in temp_traces if i[0].parent is not None]
    traces_with_daughter = [i for i in temp_traces if i[-1] in parent_cell]
    for trace in traces_with_daughter:
        daughter_traces = [i for i in temp_traces if i[0].parent is trace[-1]]
        for daughter_trace in daughter_traces:
            # To update but not replace
            daughter_trace.extend(trace)
            daughter_trace.sort(key=attrgetter('frame'))
        temp_traces.pop(temp_traces.index(trace))

    for trace in temp_traces:
        if len(trace) <= minframe:
            abs_id_list = [tmpcell.abs_id for tmpcell in trace]
            for cell in [i for j in trhandler.traces for i in j]:
                if cell.abs_id in abs_id_list:
                    cell.cell_id = 0
    return trhandler.traces


def gap_closing(traces, holder, DISPLACEMENT=100, MASSTHRES=0.15, maxgap=4):

    trhandler = TracesController(traces)
    '''
    Make sure not to have a cell as both disappeared and appeared cells
    '''
    for trace in trhandler.traces[:]:
        if len(trace) < 2:
            trhandler.traces.remove(trace)
    dist = trhandler.pairwise_dist()
    massdiff = trhandler.pairwise_mass()
    framediff = trhandler.pairwise_frame()

    withinarea = dist < DISPLACEMENT
    inmass = abs(massdiff) < MASSTHRES
    inframe = (framediff > 1) * (framediff <= maxgap)
    withinarea_inframe = withinarea * inframe * inmass
    # CHECK: distance as a fine cost
    withinarea_inframe = one_to_one_assignment(withinarea_inframe, dist)

    if withinarea_inframe.any():
        disapp_idx, app_idx = np.where(withinarea_inframe)

        dis_cells = trhandler.disappeared()
        app_cells = trhandler.appeared()
        for disi, appi in zip(disapp_idx, app_idx):
            dis_cell, app_cell = dis_cells[disi], app_cells[appi]
            dis_cell.next = app_cell

            # You can simply reconstruct the trace, but here to reduce the calculation,
            # connect them explicitly.
            dis_trace = [i for i in trhandler.traces if dis_cell in i][0]
            app_trace = [i for i in trhandler.traces if app_cell in i][0]
            dis_trace.extend(trhandler.traces.pop(trhandler.traces.index(app_trace)))
    return trhandler.traces


def detect_division(traces, holder, DISPLACEMENT=50, maxgap=4, DIVISIONMASSERR=0.15):
    '''
    Connect by assigning parent.
    I don't want to copy the parents yet due to reduce the dependency.
    '''
    trhandler = TracesController(traces)
    for trace in trhandler.traces[:]:
        if len(trace) < 2:
            trhandler.traces.remove(trace)

    dist = trhandler.pairwise_dist()
    massdiff = trhandler.pairwise_mass()
    framediff = trhandler.pairwise_frame()
    half_massdiff = massdiff + 0.5

    withinarea = dist < DISPLACEMENT
    inframe = (framediff <= maxgap) * (framediff >= 1)
    halfmass = abs(half_massdiff) < DIVISIONMASSERR

    withinarea_inframe_halfmass = withinarea * inframe * halfmass

    # CHECK: now based on distance.
    par_dau = one_to_two_assignment(withinarea_inframe_halfmass, half_massdiff)
    # CHECK: If only one daughter is found ignore it.
    par_dau[par_dau.sum(axis=1) == 1] = False

    if par_dau.any():
        disapp_idx, app_idx = np.where(par_dau)

        for disi, appi in zip(disapp_idx, app_idx):
            dis_cell = trhandler.disappeared()[disi]
            app_cell = trhandler.appeared()[appi]
            app_cell.parent = dis_cell
    # concatenate_parents_daughters()
    return trhandler.traces

# def concatenate_parents_daughters(self):
#     '''This part is little tricky, but we can concatenate the traces
#     while keeping a single reference to a parent, so when you update the cell_id of parent,
#     it will reflect to two traces... Deletion might not work well.
#     When you reconstruct the traces, you should use "previous" instead of "next".
#     '''
#     # Concatenate parent and daughters
#     traces = trhandler.traces
#     parent_cell = [i[0].parent for i in traces if i[0].parent is not None]
#     traces_with_daughter = [i for i in traces if i[-1] in parent_cell]
#     for trace in traces_with_daughter:
#         daughter_traces = [i for i in traces if i[0].parent is trace[-1]]
#         for daughter_trace in daughter_traces:
#             # To update but not replace
#             daughter_trace[0].previous = trace[-1]
#             daughter_trace.extend(trace)
#             daughter_trace.sort(key=attrgetter('frame'))
#         traces.pop(traces.index(trace))
