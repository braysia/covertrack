import segmentation_operations as operations
import png
import numpy as np
from os.path import join, basename
from logging import getLogger
from covertrack.utils.file_handling import _check_if_processed
from scipy.ndimage import imread
import json

ARG_VAR = 'segment_args'


class Holder(object):
    '''If you need to pass some extra arguments to run operations, use this class'''
    pass


class SegmentationCaller(object):
    logger = getLogger('covertrack.segmentation')
    PROCESS = 'Segmentation'
    holder = Holder()

    def __init__(self, outputdir):
        with open(join(outputdir, 'setting.json')) as file:
            self.argdict = json.load(file)
        self.argdict = _check_if_processed(self.argdict)

    def run(self):
        """
        By default, it will set object_name to 'nuclei' and ch to channels[0].
        """
        out_folder = basename(self.argdict['outputdir'])
        self.logger.warn('{0} started for {1}.'.format(self.PROCESS, out_folder))
        for func_args in self.argdict[ARG_VAR]:
            self.obj = func_args.pop('object_name') if 'object_name' in func_args else 'nuclei'
            self.ch = func_args.pop('ch_img') if 'ch_img' in func_args else self.argdict['channels'][0]
            self.func_args = func_args
            self.iter_channels()
        self.logger.info(self.PROCESS + ' completed.')

    def iter_channels(self):
        ''' no loop, just the first channel'''
        self.pathset = self.argdict['channeldict'][self.ch]
        self.iter_frames()

    def iter_frames(self):
        '''iterate over frames
        '''
        for frame, imgpath in enumerate(self.pathset):
            self.logger.info('\t{0} frame {1}...'.format(self.PROCESS, frame))
            self.img = imread(imgpath)
            self.run_operations()
            self.check_results_each_frame()
            self.save_output(imgpath)

    def run_operations(self):
        func_args = self.func_args
        func_args = func_args.copy()
        func_name = func_args.pop('name')
        func = getattr(operations, func_name)
        self.func = func
        self.label = func(self.img, self.holder, **func_args)

    def check_results_each_frame(self):
        msg = '{0} objects found...'.format(len(np.unique(self.label)-1))
        self.logger.info(msg)

    def save_output(self, imgpath):
        '''Save label in segmented folder.
        '''
        label = np.uint16(self.label)
        directory = join(self.argdict['outputdir'], 'segmented')
        filename = basename(imgpath).split('.')[0] + '_{0}.png'.format(self.obj)
        png.from_array(label, 'L').save(join(directory, filename))
