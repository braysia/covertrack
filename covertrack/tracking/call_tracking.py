import tracking_operations as operations
from scipy.ndimage import imread
import png
import numpy as np
from os.path import join, basename
from track_utils.cell_container import Container
from covertrack.cell import CellListMaker
from logging import getLogger
from covertrack.utils.file_handling import _check_if_processed
import json

ARG_VAR = 'track_args'


class Holder(object):
    '''If you need to pass some extra arguments to run operations, use this class'''
    pass


class TrackingCaller(object):
    PROCESS = 'Tracking'
    logger = getLogger('covertrack.tracking')
    holder = Holder()

    def __init__(self, outputdir):
        with open(join(outputdir, 'setting.json')) as file:
            self.argdict = json.load(file)
        self.argdict = _check_if_processed(self.argdict)
        self.container = Container(self.argdict['img_shape'])
        self.holder.max_cell_id = 0

    def run(self):
        out_folder = basename(self.argdict['outputdir'])
        self.logger.warn('{0} started for {1}.'.format(self.PROCESS, out_folder))
        self.iter_channels()
        self.logger.info(self.PROCESS + ' completed.')

    def iter_channels(self):
        ''' no loop, just the first channel'''
        fir_func_arg = self.argdict[ARG_VAR][0]
        ch = fir_func_arg.pop('ch_img') if 'ch_img' in fir_func_arg else self.argdict['channels'][0]
        self.obj = fir_func_arg.pop('object_name') if 'object_name' in fir_func_arg else 'nuclei'
        self.pathset = self.argdict['channeldict'][ch]
        self.iter_frames()

    def iter_frames(self):
        '''iterate over frames
        '''
        self.holder.pathset = self.pathset
        for frame, imgpath in enumerate(self.pathset):
            self.holder.frame = frame
            self.logger.info('\t{0} frame {1}...'.format(self.PROCESS, frame))
            self.img = imread(imgpath)
            self.label = self.load_label(imgpath)
            self.holder.img_shape = self.argdict['img_shape']
            self.holder.imgpath = imgpath
            self.prepare_curr_cells()
            if self.container.prev_cells is not None:
                self.run_operations()
            self.label_untracked()
            self.container.prev_cells = self.container.curr_cells
            self.holder.prev_img = self.img
            self.holder.prev_label = self.label
            self.save_output(imgpath)

    def load_label(self, imgpath):
        directory = join(self.argdict['outputdir'], 'segmented')
        filename = basename(imgpath).split('.')[0] + '_{0}.png'.format(self.obj)
        return imread(join(directory, filename))

    def prepare_curr_cells(self):
        self.container.curr_cells = CellListMaker(self.img, self.label, self.holder.frame).make_list()

    def label_untracked(self):
        # set cell id
        for cell in self.container.curr_cells:
            if cell.previous is None:
                self.holder.max_cell_id += 1
                cell.cell_id = self.holder.max_cell_id
            elif cell.previous is not None:
                cell.cell_id = cell.previous.cell_id

    def run_operations(self):
        for func_args in self.argdict[ARG_VAR]:
            func_args = func_args.copy()
            func_name = func_args.pop('name')
            func = getattr(operations, func_name)
            self.container = func(self.img, self.label, self.container, self.holder, **func_args)
            tres = (func.__name__, len(self.container.linked[1]), len(self.container.unlinked[1]))
            self.logger.info('\t{0}:  {1} linked, {2} more unlinked.'.format(*tres))

    def save_output(self, imgpath):
        label = np.zeros(self.argdict['img_shape'], np.uint16)
        for cell in self.container.curr_cells:
            if cell.cell_id is not None:
                label[cell.prop.coords[:, 0], cell.prop.coords[:, 1]] = cell.cell_id
        directory = join(self.argdict['outputdir'], 'tracked')
        filename = basename(imgpath).split('.')[0] + '_' + self.obj + '.png'
        png.from_array(label, 'L').save(join(directory, filename))
