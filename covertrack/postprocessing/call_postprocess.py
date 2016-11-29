import postprocessing_operations as operations
import numpy as np
from logging import getLogger
from posttrack_utils.traces import assign_next_and_abs_id_to_storage
from posttrack_utils.traces import construct_traces_based_on_next
from posttrack_utils.traces import convert_traces_to_storage
from posttrack_utils.traces import connect_parent_daughters
from posttrack_utils.postprocess_io import save_label, save_div_img
from posttrack_utils.postprocess_io import save_df, load_label
import json
from os.path import join, basename
try:
    from covertrack.cell import CellListMakerScalar
    from covertrack.utils.file_handling import _check_if_processed
except:
    from cell import CellListMakerScalar
    from utils.file_handling import _check_if_processed
from scipy.ndimage import imread
import json
from posttrack_utils.postprocess_io import make_obj_path
import time

SAVE_DIVISION = False
ARG_VAR = 'postprocess_args'

'''
traces: a list of cells connected by next, ordered by frame
'''

class Holder(object):
    '''If you need to pass some extra arguments to run operations, use this class'''
    pass


class PostprocessCaller(object):
    storage = []
    logger = getLogger('covertrack.postprocess')
    PROCESS = 'Postprocessing'
    holder = Holder()

    def __init__(self, outputdir):
        with open(join(outputdir, 'setting.json')) as file:
            self.argdict = json.load(file)
        self.argdict = _check_if_processed(self.argdict)

    def run(self):
        out_folder = basename(self.argdict['outputdir'])
        self.logger.warn('{0} started for {1}.'.format(self.PROCESS, out_folder))
        self.iter_channels()
        self.logger.info(self.PROCESS + ' completed.')

    def iter_channels(self):
        ''' no loop, just the first channel'''
        fir_func_arg = self.argdict[ARG_VAR][0]
        ch = fir_func_arg.pop('ch_img') if 'ch_img' in fir_func_arg else self.argdict['channels'][0]
        self.ch = ch
        self.obj = fir_func_arg.pop('object_name') if 'object_name' in fir_func_arg else 'nuclei'
        self.pathset = sorted(self.argdict['channeldict'][self.ch])
        self.iter_frames()
        self.post_operation()

    def iter_frames(self):
        '''iterate over frames
        '''
        self.storage = []
        self.logger.info('\t{0} , collecting cells info...'.format(self.PROCESS))
        for frame, imgpath in enumerate(self.pathset):
            # self.logger.info('\t{0} frame {1}...'.format(self.PROCESS, frame))
            self.img = imread(imgpath)
            self.label = load_label(self.argdict['outputdir'], imgpath, self.obj)
            self.holder.img_shape = self.argdict['img_shape']
            curr_cells = CellListMakerScalar(self.img, self.label,
                                             self.holder, frame).make_list()
            self.storage.append(curr_cells)

    def run_operations(self):
        self.holder.num_frame = len(self.argdict['channeldict'].values()[0])
        for func_args in self.argdict[ARG_VAR]:
            func_args = func_args.copy()
            func_name = func_args.pop('name')
            func = getattr(operations, func_name)
            self.traces = func(self.traces, self.holder, **func_args)
            self.logger.info('\t{0} done.'.format(func.__name__))

    def post_operation(self):
        # make a link between cells by assigning next
        self.storage = assign_next_and_abs_id_to_storage(self.storage)
        # make traces from storage.
        self.traces = construct_traces_based_on_next(self.storage)
        self.run_operations()
        self.logger.info('Saving labels...')
        self._save_storage_as_labels()
        self._add_objdict_to_json()
        self.logger.info('Saving initial dataframe...')
        self._save_df()

    def _save_storage_as_labels(self):
        # reconstruct traces
        storage = convert_traces_to_storage(self.traces)
        self.traces = construct_traces_based_on_next(storage)
        self.traces = connect_parent_daughters(self.traces)
        # Assign the same cell_id to each trace
        for num, trace in enumerate(self.traces):
            for cell in trace:
                if cell.cell_id != 0:  # 0 means to be removed
                    cell.cell_id = num+1
        cells = [i for j in self.traces for i in j]

        # Update label and save it in objects folder
        for frame, imgpath in enumerate(self.pathset):
            label = load_label(self.argdict['outputdir'], imgpath, self.obj)
            cells_in_frame = [i for i in cells if i.frame == frame]
            newlabel = np.zeros(label.shape, np.uint16)
            for cell in cells_in_frame:
                newlabel[label == cell.prop.label_id] = cell.cell_id
            save_label(newlabel, self.argdict['outputdir'], imgpath, self.obj)
        self.storage = cells

    def _add_objdict_to_json(self):
        objpathset = [make_obj_path(self.argdict['outputdir'], i, self.obj) for i in self.pathset]
        if 'objdict' not in self.argdict:
            self.argdict['objdict'] = {}
        self.argdict['objdict'][self.obj] = objpathset
        with open(join(self.argdict['outputdir'], 'setting.json'), 'w') as f1:
            json.dump(self.argdict, f1, indent=4)
        self.logger.info('setting.json updated with objdict')
        time.sleep(1)

    def _save_df(self):
        '''Save output as DataFrame'''
        save_df(self.storage, self.argdict['outputdir'], self.ch, self.obj)

    def last_process(self):
        '''Save images highlighting division in cleaned folder.
        '''
        if SAVE_DIVISION:
            save_div_img(self.argset, self.obj, self.pathset, self.storage)
