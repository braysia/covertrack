import subdetection_operations as operations
from scipy.misc import imread
import png
from os.path import join, basename
from logging import getLogger
from covertrack.utils.seg_utils import find_label_boundaries
from covertrack.utils.file_handling import _check_if_processed
import json
from itertools import izip_longest
from covertrack.utils.file_handling import imgread

ARG_VAR = 'subdetect_args'



class Holder(object):
    '''If you need to pass some extra arguments to run operations, use this class'''
    pass


class SubDetection(object):
    logger = getLogger('covertrack.subdetection')
    PROCESS = 'Subdetection'
    holder = Holder()

    def __init__(self, outputdir):
        with open(join(outputdir, 'setting.json')) as file:
            self.argdict = json.load(file)
        self.argdict = _check_if_processed(self.argdict)

    def run(self):
        out_folder = basename(self.argdict['outputdir'])
        self.logger.warn('{0} started for {1}.'.format(self.PROCESS, out_folder))
        for func_args in self.argdict[ARG_VAR]:
            pathset, seed_pathset = [], []
            self.obj = func_args.pop('object_name')
            if 'seed_obj' in func_args:
                if func_args['seed_obj']:
                    seed_obj = func_args.pop('seed_obj')
                    seed_pathset = self.argdict['objdict'][seed_obj]
            if 'ch_img' in func_args:
                self.channel = func_args.pop('ch_img')
                if self.channel:
                    if isinstance(self.channel, list) or isinstance(self.channel, tuple):
                        pathset = zip(*[self.argdict['channeldict'][i] for i in self.channel])
                    else:
                        pathset = self.argdict['channeldict'][self.channel]
            else:
                self.channel = self.argdict['channels'][0]
                pathset = self.argdict['channeldict'][self.argdict['channels'][0]]
            self.iter_frames(func_args, pathset, seed_pathset)

    def iter_frames(self, func_args, pathset, seed_pathset):
        for self.frame, (imgpath, seed_imgpath) in enumerate(izip_longest(pathset, seed_pathset)):
            self.logger.info('\t{0} frame {1}...'.format(self.PROCESS, self.frame))
            self.img = imgread(imgpath) if imgpath else None
            self.label = self._load_label(seed_imgpath) if seed_imgpath else None
            self.holder.imgpath = imgpath
            self.holder.seed_imgpath = seed_imgpath
            self.holder.argdict = self.argdict
            self.run_operations(func_args)
            if isinstance(imgpath, list) or isinstance(imgpath, tuple):
                imgpath = imgpath[0]
            self.save_output(imgpath)

    def run_operations(self, func_args):
        func_args = func_args.copy()
        func_name = func_args.pop('name')
        func = getattr(operations, func_name)
        self.sub_label = func(self.img, self.label, self.holder, **func_args)

    def _load_label(self, imgpath):
        return imread(imgpath)

    def save_output(self, imgpath):
        directory = join(self.argdict['outputdir'], 'objects')
        filename = basename(imgpath).split('.')[0] + '_{0}.png'.format(self.obj)
        png.from_array(self.sub_label, 'L').save(join(directory, filename))
        directory = join(self.argdict['outputdir'], 'outlines')
        filename = basename(imgpath).split('.')[0] + '_{0}_outlines.png'.format(self.obj)
        png.from_array(find_label_boundaries(self.sub_label), 'L').save(join(directory, filename))
