import numpy as np
import pandas as pd
from os.path import join, dirname
from scipy.io import savemat


def add_median_ratio_cytoplasm_nuclei(df):
    '''Add median_ratio in DataFrame if it has nuclei and cytoplasm objects.
    nuc/cyto is stored in nuclei, cyto/nuc is stored in cytoplasm
    '''
    object_names = np.unique(df.index.get_level_values('object'))
    if 'cytoplasm' in object_names and 'nuclei' in object_names:
        nuc = df.xs(['nuclei', 'median_intensity'], level=['object', 'prop'])
        cyto = df.xs(['cytoplasm', 'median_intensity'], level=['object', 'prop'])
        median_ratio_cyto = organize_index(cyto/nuc, 'cytoplasm')
        median_ratio_nuc = organize_index(nuc/cyto, 'nuclei')
        df = pd.concat([df, median_ratio_nuc, median_ratio_cyto])
    return df

def add_median_ratio_cytoplasm_nuclei_old(arr, labels):
    '''Add median_ratio in DataFrame if it has nuclei and cytoplasm objects.
    nuc/cyto is stored in nuclei, cyto/nuc is stored in cytoplasm
    '''

    cytomedian = [a for num, a in enumerate([i for i in labels if len(i) == 3])
                  if 'cytoplasm' in a[0] and 'median_intensity' in a[2]]
    nucmedian = [a for num, a in enumerate([i for i in labels if len(i) == 3])
                 if 'nuclei' in a[0] and 'median_intensity' in a[2]]
    fields = [(i, ii) for i, ii in zip(cytomedian, nucmedian) if i[1] == ii[1]]
    for ci, ni in fields:
        template = np.zeros((2, arr.shape[1], arr.shape[2]), np.float32)
        template[0, :, :] = arr[labels.index(ci), :, :]/arr[labels.index(ni), :, :]
        template[1, :, :] = arr[labels.index(ni), :, :]/arr[labels.index(ci), :, :]
        arr = np.concatenate((template, arr), axis=0)
        new_labels = [('cytoplasm', ci[1], 'median_ratio'), ('nuclei', ci[1], 'median_ratio')]
        labels = new_labels + labels
    return arr, labels


def organize_index(median_ratio, object_name):
    median_ratio['object'] = object_name
    median_ratio['prop'] = 'median_ratio'
    median_ratio.set_index('object', append=True, inplace=True)
    median_ratio.set_index('prop', append=True, inplace=True)
    median_ratio = median_ratio.reorder_levels(['object', 'ch', 'prop', 'frame'])
    return median_ratio


def df_to_mat(dfpath):
    '''
    Save mat file for GUI. This will be updated and removed in the future.
    imgpahts is np.array with rows corresponding to channels and columns corresponding to frames. array of basenames.
    '''
    df = pd.read_csv(dfpath, index_col=['object', 'ch', 'prop', 'frame'])
    data = {}
    data['subfolderMetadata'] = {}
    data['subfolderMetadata']['subfolderName'] = 'PosExample'
    data['inputStruct'] = {}
    data['inputStruct']['cpDataFilename'] = 'cpData.mat'
    object_names = list(set(df.index.get_level_values('object')))
    channels = list(set(df.index.get_level_values('ch')))
    props = list(set(df.index.get_level_values('prop')))
    for i in object_names:
        data[i] = {}
        data[i]['imageSetIdx'] = np.arange(len(channels))[:, np.newaxis]+1
        data[i]['imageSetNames'] = np.zeros((len(channels),1), dtype='object')
        for n, chi in enumerate(channels):
            data[i]['imageSetNames'][n] = str(chi)
        for pi in props:
            # FIXME: check if matrix size is identical
            if pi in ['x', 'y']:
                data[i][pi] = {}
                data[i][pi] = np.array(df.ix[i, channels[0], pi]).T
            else:
                data[i][pi] = {}
                try:
                    data[i][pi] = np.array(np.dstack([np.array(df.ix[i, chi, pi]).T for chi in channels]))
                except:
                    import ipdb;ipdb.set_trace()
        data[i]['label'] = np.array(df.ix[object_names[0], channels[0], 'label_id'].copy()).T
    data['imageSetChannels'] = np.zeros((len(channels),1), dtype='object')
    for i, chi in enumerate(channels):
        data['imageSetChannels'][i] = chi
    data['objectSetNames'] = np.zeros((len(object_names), 1), dtype='object')
    for n, objn in enumerate(object_names):
        data['objectSetNames'][n] = objn

    outputpath = dirname(dfpath)
    try:
        imgpaths = np.load(join(outputpath, 'imgpaths.npy'))
        data['imageSetFilenames'] = imgpaths
    except:
        print "imgpaths.npy not saved in the same output directory"
    data['frames'] = np.arange(data[object_names[0]][props[0]].shape[1], dtype=np.float64)

    store = {}
    store['data'] = data
    savemat(join(outputpath, 'cpDataTracked.mat'), store)
