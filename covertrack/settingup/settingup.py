import os
import types
from os.path import join, basename, exists
import imp
import json
from scipy.ndimage import imread
import time
from logging import getLogger
import setup_operations as operations


# Output folders
FOLDERNAMES = ['objects', 'channels', 'outlines', 'storage', 'processed',
               'segmented', 'tracked', 'cleaned']


class Holder(object):
    '''If you need to pass some extra arguments to run operations, use this class'''
    pass


class SettingUpCaller(object):
    PROCESS = "SettingUp"
    logger = getLogger('covertrack.setup')
    holder = Holder()

    def __init__(self, ia_path='../input_file/input_tests1.py', imgdir=None):
        self.ia_path = ia_path
        self.argfile = imp.load_source('inputArgs', ia_path)
        self.imgdir = imgdir
        self.argdict = {}

    def run(self):
        self.set_explicit_args()
        self.prepare_outputdir()
        self.run_operations()
        self._load_img_shape()
        self._set_time()
        self.save_setting()
        self.logger.warn('{0} completed for {1}.'.format(self.PROCESS, self.argdict['outputdir']))
        return self.argdict['outputdir']

    def run_operations(self):
        for func_args in self.argdict['setup_args']:
            func_args = func_args.copy()
            func_name = func_args.pop('name')
            func = getattr(operations, func_name)
            self.argdict = func(self.argdict, self.imgdir, self.holder, **func_args)

    def _load_img_shape(self):
        self.argdict['img_shape'] = imread(self.argdict['channeldict'].values()[0][0]).shape

    def _set_time(self):
        if 'time' not in self.argdict:
            self.argdict['time'] = range(len(self.argdict['channeldict'].values()[0]))

    def save_setting(self):
        with open(join(self.argdict['outputdir'], 'setting.json'), 'w') as f1:
            json.dump(self.argdict, f1, indent=4)
        time.sleep(1)

    def set_explicit_args(self):
        '''Set input arguments to ArgSet attributes.
        If imgdir is not passed to SettingUp, it will set input_parent_dir
        as imgdir.
        '''
        ia_args = [a for a in dir(self.argfile) if not a.startswith('_')]
        ia_args = [a for a in ia_args if not isinstance(getattr(self.argfile, a), types.ModuleType)]
        ia_args = [a for a in ia_args if not isinstance(getattr(self.argfile, a), types.FunctionType)]
        for a in ia_args:
            self.argdict[a] = getattr(self.argfile, a)
        if self.imgdir is None:
            self.imgdir = self.argdict['input_parent_dir']

    def prepare_outputdir(self):
        '''set {output_parent_dir}/{foldername} as outputdir and make a folder.
        '''
        foldername = basename(self.imgdir)
        self.argdict['outputdir'] = join(self.argdict['output_parent_dir'], foldername)
        if not exists(self.argdict['outputdir']):
            os.makedirs(self.argdict['outputdir'])
        self.__prepare_outputdir_subfolders()

    def __prepare_outputdir_subfolders(self):
        '''make subfolders under self.argset.outputdir
        '''
        for fn in FOLDERNAMES:
            if not exists(join(self.argdict['outputdir'], fn)):
                os.makedirs(join(self.argdict['outputdir'], fn))

if __name__ == "__main__":
    caller = SettingUpCaller()
