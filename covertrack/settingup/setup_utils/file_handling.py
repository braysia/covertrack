import os
from os.path import join, exists
import numpy as np
import pandas as pd


class MakeDataFrame(object):
    def __init__(self, storage, outputdir, channel, object_name):
        self.storage = storage
        self.outputdir = outputdir
        self.ch = channel
        self.obj = object_name

    def save(self):
        self._initialize_data()

    def _initialize_data(self):
        self.df = make_multi_index_pandas(self.storage, self.obj, self.ch)
        return self.df


class MakeDataFrameSelected(MakeDataFrame):
    def _initialize_data(self):
        selections = ['area', 'eccentricity', 'frame', 'label_id', 'max_intensity', 'mean_intensity', 'median_intensity', 'mergeto', 'min_intensity', 'perimeter', 'splitfrom', 'total_intensity', 'x', 'y', 'abs_id', 'parent_id']
        keys = [j for j in dir(self.storage[0]) if not j.startswith('_')]
        selected_keys = [i for i in keys if i in selections]
        self.df = make_multi_index_pandas(self.storage, self.obj, self.ch, selected_keys)
        return self.df


def make_multi_index_pandas(storage, object_name, channel, keys=[]):
    cell_ids = np.unique([i.label_id for i in storage])
    frames = np.unique([i.frame for i in storage])

    if not keys:
        keys = [j for j in dir(storage[0]) if not j.startswith('_')]
    keys_frames = [['{0}__{1}'.format(k, i) for i in frames] for k in keys]
    keys_frames = [i for ii in keys_frames for i in ii]
    index = pd.MultiIndex.from_product([object_name, channel, keys, frames], names=['object', 'ch', 'prop', 'frame'])
    # column_idx = pd.MultiIndex.from_product([cell_ids], names=['id'])
    column_idx = pd.MultiIndex.from_product([cell_ids])
    df = pd.DataFrame(index=index, columns=column_idx)
    for cell in storage:
        for k in keys:
            df[cell.label_id].loc[object_name, channel, k, cell.frame] = getattr(cell, k)
    return df


class ImgdirsFinder():
    def __init__(self, parentdir):
        self.parentdir = parentdir

    def find_dirs(self):
        self.find_dirs_with_metadatxt()
        if not self.imgdirs:
            self.find_dirs_with_images()
        if not self.imgdirs:
            raise Exception('imgdirs do not exist')
        return self.imgdirs


    def find_dirs_with_metadatxt(self):
        imgdirs = []
        for root, dirs, files in os.walk(self.parentdir):
            if 'metadata.txt' in files:
                imgdirs.append(root)
        self.imgdirs = imgdirs

    def find_dirs_with_images(self):
        imgdirs = []
        for root, dirs, files in os.walk(self.parentdir):
            if check_if_imgdir(root):
                imgdirs.append(root)
        self.imgdirs = imgdirs


def check_if_imgdir(path):
        imgs = [i for i in os.listdir(path) if ('.png' in i) or ('.tiff' in i)]
        if imgs:
            return True
        else:
            return False


def check_if_images_in_imgpaths_dict(f):
    def wrapper(self):
        if not check_if_imgdir(self.imgdir):
            self.logger.warn('images not found in imgdir.')
            raise Exception
        f(self)
        len_imgs = [len(i) for i in self.imgpaths_dict.itervalues()]
        if len_imgs[0]==0:
            self.logger.warn('channels not found in imgdir?')
            raise Exception
        if not all([x == len_imgs[0] for x in len_imgs]):
            self.logger.warn('Each channels have different number of images?')
            raise Exception
    return wrapper
