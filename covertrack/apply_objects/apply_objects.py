from scipy.misc import imread
from os.path import join, basename
from covertrack.cell import CellListMakerScalar
from covertrack.utils.file_handling import ConvertDfSelected2 as ConvertDfSelected
import os
import pandas as pd
import numpy as np
from covertrack.utils.df_handling import add_median_ratio_cytoplasm_nuclei
from os.path import join
from logging import getLogger
from covertrack.utils.df_handling import df_to_mat
from covertrack.utils.file_handling import _check_if_processed, pd_array_convert_cell_prop
from itertools import product
import json
import re
from covertrack.utils.file_handling import initialize_threed_array, extend_threed_array
from covertrack.utils.file_handling import save_arr_labels


# ARG_VAR = 'apply_operations.py'


class Holder(object):
    '''If you need to pass some extra arguments to run operations, use this class'''
    pass


class ApplyObjects(object):
    algs = []
    storage = []
    df = None
    logger = getLogger('covertrack.apply_objects')
    PROCESS = 'ApplyObjects'
    holder = Holder()

    def __init__(self, outputdir):
        with open(join(outputdir, 'setting.json')) as file:
            self.argdict = json.load(file)
        self.argdict = _check_if_processed(self.argdict)

    def run(self):
        out_folder = basename(self.argdict['outputdir'])
        self.logger.warn('{0} started for {1}.'.format(self.PROCESS, out_folder))

        self._load_df()
        objects, self.objdict = self.extract_object_names()
        product_ch_obj = [i for i in product(objects, self.argdict['channels']) if i[0] is not None]
        for self.obj, self.ch in product_ch_obj:
            already_existed = self._check_already_existed()
            if already_existed:
                self.logger.info('{0} in {1} already exists.'.format(self.obj, self.ch))
            else:
                self.run_each_channels_and_objects()
        self.last_process()

    def _load_df(self):
        self.df = pd.read_csv(join(self.argdict['outputdir'], 'ini_df.csv'), index_col=['object', 'ch', 'prop', 'frame'])
        self.df.columns = pd.to_numeric(self.df.columns)
        self.df = self.df.astype(np.float32)
        self.unique_frames = self.df.index.levels[3].tolist()

    def extract_object_names(self):
        obj_dir = join(self.argdict['outputdir'], 'objects')
        files = os.listdir(obj_dir)
        re_lists = [re.search('.*_(?P<f>.*).png', i) for i in files]
        strings = [i.string for i in re_lists]
        object_names = [i.group('f') for i in re_lists]
        objects = list(set(object_names))
        dic = {}
        for ob in objects:
            dic[ob] = [join(obj_dir, i) for i, ii in zip(strings, object_names) if ii == ob]
        return objects, dic

    def run_each_channels_and_objects(self):
        storage = []
        pathset = sorted(self.argdict['channeldict'][self.ch])
        object_pathset = sorted(self.objdict[self.obj])
        for frame, (imgpath, obj_imgpath) in enumerate(zip(pathset, object_pathset)):
            self.logger.info('\t frame {0}: apply {1} to {2}'.format(frame, self.obj, self.ch))
            img = imread(imgpath)
            label = imread(obj_imgpath)
            curr_cells = CellListMakerScalar(img, label, self.holder, frame).make_list()
            storage.extend(curr_cells)
        for cell in storage:  # maybe removed in the future?
            cell.cell_id = cell.prop.label_id
        [setattr(si, 'abs_id', i+1) for i, si in enumerate(storage)]
        if storage:
            df = ConvertDfSelected(storage, self.argdict['outputdir'], self.ch, self.obj, self.unique_frames)._initialize_data()
            if self.df is not None:
                df = pd.concat([df, self.df])
            self.df = df
        del storage

    def _save_pathset_and_mat(self):
        ch = self.argdict['channels']
        patharray = np.zeros((len(ch), len(self.argdict['channeldict'].values()[0])), dtype=np.object)
        for n, chi in enumerate(ch):
            patharray[n, :] = np.array([basename(i) for i in self.argdict['channeldict'][chi]]).T
        np.save(join(self.argdict['outputdir'], 'pathset'), patharray)
        self.logger.info('pathset saved')
        # df_to_mat(join(self.argdict['outputdir'], 'df.csv'))
        # self.logger.info('cpDataTracked.mat saved')

    def last_process(self):
        self.df = add_median_ratio_cytoplasm_nuclei(self.df)
        self._save_df()
        self._save_pathset_and_mat()

    def _save_df(self):
        '''To read, use df = pd.read_csv('df.csv', index_col=['object', 'ch', 'prop', 'frame'])'''
        self.df.to_csv(join(self.argdict['outputdir'], 'df.csv'))
        self.logger.info('df.csv saved')
        pd_array_convert_cell_prop(join(self.argdict['outputdir'], 'df.csv'))

    def _check_already_existed(self):
        already_existed = False
        object_existed = self.obj in self.df.index.levels[0]
        object_channel_existed = (object_existed) and (self.ch in self.df.index.levels[1])
        if object_channel_existed:
            already_existed = not (self.df.xs([self.obj, self.ch],
                                   level=['object', 'ch']).empty)
        return already_existed
