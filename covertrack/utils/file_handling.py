import os
from os.path import join, exists, dirname
import numpy as np
import pandas as pd
from os.path import basename
from itertools import product
from scipy.ndimage import imread


IMG_EXT = ['png', 'tif']
selections = ['area', 'eccentricity', 'frame', 'cell_id', 'max_intensity',
              'mean_intensity', 'median_intensity', 'mergeto', 'min_intensity',
              'perimeter', 'splitfrom', 'total_intensity', 'x', 'y', 'abs_id',
              'parent_id', 'std_intensity', 'major_axis_length', 'minor_axis_length',
              'solidity', 'convex_area', 'cv_intensity', 'abs_id', 'cell_id', 'frame']  # These properties will be saved.
cell_prop = ('parent_id', 'abs_id', 'cell_id', 'frame')


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


class ConvertDfSelected(MakeDataFrame):
    def _initialize_data(self):
        keys = [j for j in dir(self.storage[0].prop) if not j.startswith('_')]
        selected_keys = [i for i in keys if i in selections]
        self.df = make_multi_index_pandas(self.storage, self.obj, self.ch, selected_keys)
        return self.df


class ConvertDfSelected2(MakeDataFrame):
    def __init__(self, storage, outputdir, channel, object_name, unique_frames):
        self.storage = storage
        self.outputdir = outputdir
        self.ch = channel
        self.obj = object_name
        self.unique_frames = unique_frames

    def _initialize_data(self):
        keys = [j for j in dir(self.storage[0].prop) if not j.startswith('_')]
        selected_keys = [i for i in keys if i in selections]
        self.df = make_multi_index_pandas2(self.storage, self.obj, self.ch, self.unique_frames, selected_keys)
        return self.df


def make_multi_index_pandas2(storage, object_name, channel, unique_frames, keys=[]):
    cell_ids = np.unique([i.cell_id for i in storage])
    frames = np.unique(unique_frames)
    if not keys:
        keys = [j for j in dir(storage[0].prop) if not j.startswith('_')]
    keys_frames = [['{0}__{1}'.format(k, i) for i in frames] for k in keys]
    keys_frames = [i for ii in keys_frames for i in ii]
    index = pd.MultiIndex.from_product([object_name, channel, keys, frames], names=['object', 'ch', 'prop', 'frame'])
    # column_idx = pd.MultiIndex.from_product([cell_ids], names=['id'])
    column_idx = pd.MultiIndex.from_product([cell_ids])
    df = pd.DataFrame(index=index, columns=column_idx, dtype=np.float32)
    for cell in storage:
        for k in keys:
            df[cell.cell_id].loc[object_name, channel, k, cell.frame] = np.float32(getattr(cell.prop, k))
    return df



def make_multi_index_pandas(storage, object_name, channel, keys=[]):
    cell_ids = np.unique([i.cell_id for i in storage])
    frames = np.unique([i.frame for i in storage])
    if not keys:
        keys = [j for j in dir(storage[0].prop) if not j.startswith('_')]
    keys_frames = [['{0}__{1}'.format(k, i) for i in frames] for k in keys]
    keys_frames = [i for ii in keys_frames for i in ii]
    index = pd.MultiIndex.from_product([object_name, channel, keys, frames], names=['object', 'ch', 'prop', 'frame'])
    # column_idx = pd.MultiIndex.from_product([cell_ids], names=['id'])
    column_idx = pd.MultiIndex.from_product([cell_ids])
    df = pd.DataFrame(index=index, columns=column_idx, dtype=np.float32)
    for cell in storage:
        for k in keys:
            df[cell.cell_id].loc[object_name, channel, k, cell.frame] = np.float32(getattr(cell.prop, k))
    return df


def initialize_threed_array(storage, object_name, channel, cell_ids=[]):
    if not cell_ids:
        cell_ids = np.unique([i.cell_id for i in storage]).tolist()
    frames = np.unique([i.frame for i in storage]).tolist()

    keys = [j for j in dir(storage[0].prop) if not j.startswith('_')]
    keys = [i for i in keys if i in selections]

    labels = [(str(object_name), str(channel), str(k)) for k in keys]

    for i in cell_prop:
        labels.append((i, ))
    arr = np.zeros((len(keys) + len(cell_prop), len(cell_ids), len(frames)), np.float32)
    for cell in storage:
        for num, k in enumerate(keys):
            arr[num, cell_ids.index(cell.cell_id), frames.index(cell.frame)] = getattr(cell.prop, k)
        for n, i in enumerate(range(-len(cell_prop), 0, 1)):
            arr[i, cell_ids.index(cell.cell_id), frames.index(cell.frame)] = getattr(cell, cell_prop[n])
    return arr, labels


def extend_threed_array(arr, labels, storage, object_name, channel):
    cell_ids = [i for i in np.unique(arr[-2, :, :]).tolist() if i != 0]
    new_arr, new_labels = initialize_threed_array(storage, object_name, channel, cell_ids)

    temp_labels = new_labels[:-len(cell_prop)]
    template = np.zeros((len(temp_labels), arr.shape[1], arr.shape[2]), np.float32)
    for ri in range(new_arr.shape[1]):
        for ci in range(new_arr.shape[2]):
            cell_id, frame = new_arr[-2, ri, ci], new_arr[-1, ri, ci]
            bool_arr = ((arr[-2, :, :] == cell_id) * (arr[-1, :, :] == frame))
            template[:, bool_arr] = new_arr[:-len(cell_prop), bool_arr]
    arr = np.concatenate((template, arr), axis=0)
    labels = temp_labels + labels
    return arr, labels


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


def _find_img_files(imgdir):
    '''return all the files in imgdir if it has IMG_EXT.
    '''
    imglists = []
    for ext in IMG_EXT:
        imglists.extend([i for i in os.listdir(imgdir) if ext in i])
    return imglists


def _check_if_processed(argdict):
    '''For each channels, if the same number of images are found in processed folder,
    replace imgdir to processed folder.
    '''
    processed_dir = join(argdict['outputdir'], 'processed')
    processed_imgs = _find_img_files(processed_dir)
    for channel, pathset in argdict['channeldict'].iteritems():
        original_names = [basename(i) for i in pathset]
        new_names = sorted([i for i in processed_imgs if channel in i])
        if len(original_names) == len(new_names):
            argdict['channeldict'][channel] = [join(processed_dir, i) for i in new_names]
    return argdict


def pd_array_convert(path):
    df = pd.read_csv(path, index_col=['object', 'ch', 'prop', 'frame'])
    objects, channels, props = [list(i) for i in df.index.levels[:3]]
    labels = [i for i in product(objects, channels, props)]
    storage = []
    for i in labels:
        storage.append(np.float32(df.ix[i]).T)
    arr = np.rollaxis(np.dstack(storage), 2)

    dic_save = {'data': arr, 'labels': labels}
    file_name = basename(path).split('.')[0]
    np.savez_compressed(join(dirname(path), file_name), **dic_save)


def pd_array_convert_cell_prop(path):
    df = pd.read_csv(path, index_col=['object', 'ch', 'prop', 'frame'])
    objects, channels, props = [list(i) for i in df.index.levels[:3]]
    labels = [i for i in product(objects, channels, props)]
    # labels = [list(i) for i in labels]

    cell_props = [i for i in labels if i[2] in cell_prop]

    cell_props_exists = [pi for pi in cell_props if df.loc[pi].any().any()]
    cell_props_nan = [pi for pi in cell_props if not df.loc[pi].any().any()]
    for pin in cell_props_nan:
        labels.remove(pin)

    for cp in cell_prop:
        if not [a for a in cell_props_exists if cp in a]:
            cell_props_exists.append([a for a in cell_props if cp in a][0])
            labels.append([a for a in cell_props if cp in a][0])

    storage = []
    for i in labels:
        storage.append(np.float32(df.ix[i]).T)
    arr = np.rollaxis(np.dstack(storage), 2)

    new_labels = [list(i) for i in labels]
    for cpe in cell_props_exists:
        new_labels[labels.index(cpe)] = [cpe[2], ]

    dic_save = {'data': arr, 'labels': new_labels}
    file_name = basename(path).split('.')[0]
    np.savez_compressed(join(dirname(path), file_name), **dic_save)



def array_pd_convert(arr, labels):
    slabels = [list(i) for i in labels if len(i) == 3]
    object_name = set([i[0] for i in slabels])
    channels = set(i[1] for i in slabels)
    keys = set(i[2] for i in slabels)
    frames = range(arr.shape[2])
    index = pd.MultiIndex.from_product([object_name, channels, keys, frames], names=['object', 'ch', 'prop', 'frame'])
    # column_idx = pd.MultiIndex.from_product([cell_ids], names=['id'])
    column_idx = pd.MultiIndex.from_product([[int(i) for i in np.unique(arr[-2, :, :]).tolist() if i!=0]])
    df = pd.DataFrame(index=index, columns=column_idx)
    for ridx in range(arr.shape[1]):
        cell_id = int(arr[-2, ridx, :].max())
        for num, sl in enumerate(slabels):
            df[cell_id].loc[tuple(sl)] = arr[num, ridx, :]
    return df


def save_arr_labels(arr, labels, outputdir, file_name):
    dic_save = {'data': arr, 'labels': labels}
    np.savez_compressed(join(outputdir, file_name), **dic_save)


def imgread(path):
    """If path is a list, then it will stack them as a 3D np.array"""
    if isinstance(path, list) or isinstance(path, tuple):
        store = []
        for p in path:
            store.append(imread(p))
        return np.dstack(store)
    if isinstance(path, str) or isinstance(path, unicode):
        return imread(path)
