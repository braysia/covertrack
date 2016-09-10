import os
import types
from os.path import join, basename, exists
import imp
from logging import getLogger, StreamHandler, FileHandler, DEBUG, WARNING
from setup_utils.nikon_metadata_handling import NikonMetadata
from abc import ABCMeta, abstractmethod
import numpy as np
from covertrack.utils.file_handling import _find_img_files
from collections import OrderedDict
import json
from scipy.ndimage import imread
import time
from logging import getLogger


# Output folders
FOLDERNAMES = ['objects', 'channels', 'outlines', 'storage', 'processed',
               'segmented', 'tracked', 'cleaned']
IMG_EXT = ['png', 'tif']

IGNORE_PROCESSED = False


class SettingUpCaller(object):
    PROCESS = "SettingUp"
    logger = getLogger('covertrack.setup')

    def __init__(self, ia_path='../input_file/input_tests1.py', imgdir=None):
        self.ia_path = ia_path
        self.argfile = imp.load_source('inputArgs', ia_path)
        self.imgdir = imgdir

    def run(self):
        if not hasattr(self.argfile, 'setup_args'):
            outputdir = SettingUpCovert(self.imgdir, self.ia_path).implement()
        if hasattr(self.argfile, 'setup_args'):
            if self.argfile.setup_args == 'SettingUpCovert2':
                outputdir = SettingUpCovert2(self.imgdir, self.ia_path).implement()

        out_folder = basename(outputdir)
        self.logger.warn('{0} completed for {1}.'.format(self.PROCESS, out_folder))

        return outputdir


class SettingUp(object):
    '''Storage of parameters used for analysis.
    '''
    # img_shape = []
    # channels, binning, time = [], [], []
    logger = getLogger('covertrack.settingup')
    __metaclass__ = ABCMeta

    def __init__(self, imgdir=None, ia_path='./covertrack/inputArgs.py'):
        self.imgdir = imgdir
        self.argfile = imp.load_source('inputArgs', ia_path)
        self.argdict = {}

    def implement(self):
        self.set_explicit_args()
        self.prepare_outputdir()
        self.extract_channeldict()
        self.extract_metadata()
        self.argdict['img_shape'] = imread(self.argdict['channeldict'].values()[0][0]).shape
        with open(join(self.argdict['outputdir'], 'setting.json'), 'w') as f1:
            json.dump(self.argdict, f1, indent=4)
        time.sleep(1)
        return self.argdict['outputdir']

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

    @abstractmethod
    def extract_channeldict(self):
        '''Set self.argset.channeldict. channeldict is a list of Channel object.
        '''
        pass

    def extract_metadata(self):
        '''This method can be skipped.
        You can instead set these attributes in input argument or simply not
        use any algorithms that which requires metadata.
        Micromanager provides magnification, img_shape, time and binning
        in its metadata.txt.

        '''
        pass

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


class SettingUpCovert(SettingUp):

    METAFIELDS = ['magnification', 'img_shape', 'time', 'binning']

    def extract_channeldict(self):
        imglists = _find_img_files(self.imgdir)
        channeldict = OrderedDict()
        for ch in self.argdict['channels']:
            pathset = self._extract_pathset(self.imgdir, imglists, ch)
            channeldict[ch] = pathset
        self.argdict['channeldict'] = channeldict

    def _match_file_contents(imgnames, directory):
        exists = []
        for i in imgnames:
            exists.append(os.path.exists(join(directory, i)))
        return np.array(exists).any()

    def _extract_pathset(self, imgdir, imglists, ch):
        pathset = [join(imgdir, i) for i in imglists if ch in i]
        pathset = sorted(pathset)[self.argdict['first_frame']:self.argdict['last_frame']]
        return pathset

    def extract_metadata(self):
        pass
        # nmd = NikonMetadata(self.argdict, self.imgdir)
        # for fi in self.METAFIELDS:
        #     if not self.argdict.has_key(fi):
        #         func = getattr(nmd, '_extract_'+fi)
        #         self.argdict[fi] = func()


class SettingUpCovert2(SettingUpCovert):
    METAFIELDS = ['magnification', 'img_shape', 'time', 'binning']

    def extract_channeldict(self):
        imglists = _find_img_files(self.imgdir)
        channeldict = OrderedDict()
        for num, ch in enumerate(self.argdict['channels']):
            pathset = self._extract_pathset(self.imgdir, imglists, ch, self.argdict['chnum'][num])
            channeldict[ch] = pathset
        self.argdict['channeldict'] = channeldict


    def _extract_pathset(self, imgdir, imglists, ch, chnum):
        pathset = [join(imgdir, i) for i in imglists if 'channel{0}'.format(chnum) in i]
        pathset = sorted(pathset)[self.argdict['first_frame']:self.argdict['last_frame']]
        return pathset

if __name__ == "__main__":
    caller = SettingUpCaller()
