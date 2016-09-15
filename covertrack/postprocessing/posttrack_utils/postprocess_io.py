from os.path import join, basename
import png
from traces import division_frames_and_cell_ids
import numpy as np
from scipy.misc import imread
from skimage.io import imsave
from covertrack.utils.file_handling import initialize_threed_array
from covertrack.utils.file_handling import ConvertDfSelected
from covertrack.utils.seg_utils import find_label_boundaries


def make_obj_path(outputdir, imgpath, object_name, folder='objects', apd=''):
    directory = join(outputdir, folder)
    filename = basename(imgpath).split('.')[0] + '_{0}{1}.png'
    filename = filename.format(object_name, apd)
    return join(directory, filename)


def save_label(label, outputdir, imgpath, obj_name):
    # Save objects
    obj_path = make_obj_path(outputdir, imgpath, obj_name)
    png.from_array(label, 'L').save(obj_path)
    # Save outlines
    outline_path = make_obj_path(outputdir, imgpath, obj_name, folder='outlines', apd='_outlines')
    outline = find_label_boundaries(label)
    png.from_array(outline, 'L').save(outline_path)


def save_div_img(argset, obj, pathset, storage):
    ch_directory = join(argset.outputdir, 'channels')
    label_directory = join(argset.outputdir, 'outlines')
    object_name = obj
    img_paths = pathset
    img_names = [basename(i).split('.')[0] for i in img_paths]
    div_frame, divied_cell_ids = division_frames_and_cell_ids(storage)
    for frame, img_name in enumerate(img_names):
        img = imread(join(ch_directory, img_name+'.jpg'))
        img_name = img_name+'_{0}_outlines.png'.format(object_name)
        label = imread(join(label_directory, img_name))
        template = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        div_template = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        for divf, cell in zip(div_frame, divied_cell_ids):
            if cell == 0:
                pass
            else:
                if frame == divf:
                    div_template[label == cell] = 255
                else:
                    template[label == cell] = 255
        newlabel = np.dstack([img, template, div_template])
        directory = join(argset.outputdir, 'cleaned')
        imsave(join(directory, img_name+'.jpg'), newlabel)


def load_label(outputdir, imgpath, obj_name):
    directory = join(outputdir, 'tracked')
    filename = '{0}_{1}.png'.format(basename(imgpath).split('.')[0], obj_name)
    return imread(join(directory, filename))


def save_df(storage, outputdir, ch, obj):
    for cell in storage:
        if cell.parent is not None:
            # FIXME: ?
            cell.prop.parent_id = cell.parent.cell_id
    for cell in storage:
        cell.prop.frame = cell.frame
        cell.prop.abs_id = cell.abs_id
        cell.prop.cell_id = cell.cell_id
    # Ignore cells with cell_id == 0
    storage = [i for i in storage if i.cell_id != 0]
    df = ConvertDfSelected(storage, outputdir, ch, obj)._initialize_data()
    df.to_csv(join(outputdir, 'ini_df.csv'))


def save_df_old(storage, outputdir, ch, obj):
    for cell in storage:
        if cell.parent is not None:
            # FIXME: ?
            cell.parent = cell.parent.cell_id
    # Ignore cells with cell_id == 0
    storage = [i for i in storage if i.cell_id != 0]
    arr, labels = initialize_threed_array(storage, obj, ch)
    dic_save = {'data': arr, 'labels': labels}
    np.savez_compressed(join(outputdir, 'ini_df'), **dic_save)
