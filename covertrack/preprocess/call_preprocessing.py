import preprocess_operations as operations
import png
import numpy as np
from os.path import join, basename
from logging import getLogger
import json
from scipy.ndimage import imread


ARG_VAR = 'preprocess_args'


class Holder(object):
    '''If you need to pass some extra arguments to run operations, use this class'''
    pass


class PreprocessCaller(object):
    '''This module will read images from imgdir and save processed images in
    processed folder.
    '''
    logger = getLogger('covertrack.preprocessing')
    PROCESS = 'Preprocessing'
    holder = Holder()

    def __init__(self, outputdir):
        with open(join(outputdir, 'setting.json')) as file:
            self.argdict = json.load(file)

    def run(self):
        out_folder = basename(self.argdict['outputdir'])
        self.logger.warn('{0} started for {1}.'.format(self.PROCESS, out_folder))
        self.iter_channels()
        self.logger.info(self.PROCESS + ' completed.')

    def iter_channels(self):
        '''iterate over channels.
        '''
        for self.ch in self.argdict['channels']:
            self.pathset = self.argdict['channeldict'][self.ch]
            self.iter_frames()

    def iter_frames(self):
        '''iterate over frames
        '''
        for frame, imgpath in enumerate(self.pathset):
            self.logger.info('\t{0} frame {1}...'.format(self.PROCESS, frame))
            self.img = imread(imgpath).astype(np.uint16)
            self.run_operations()
            self.save_output(imgpath)

    def run_operations(self):
        for func_args in self.argdict[ARG_VAR]:
            func_args = func_args.copy()
            func_ch = func_args.pop('ch') if 'ch' in func_args else self.ch
            if self.ch != func_ch:
                    pass
            else:
                func_name = func_args.pop('name')
                func = getattr(operations, func_name)
                self.img = func(self.img, self.holder, **func_args)
                self.logger.info('\t{0} applied to {1}'.format(func_name, self.ch))

    def save_output(self, imgpath):
        '''Save img as output.
        '''
        label = np.uint16(self.img)
        directory = join(self.argdict['outputdir'], 'processed')
        filename = basename(imgpath).split('.')[0] + '.png'
        png.from_array(label, 'L').save(join(directory, filename))
