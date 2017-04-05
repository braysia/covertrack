from os.path import dirname, abspath, join
import sys
try:
    from covertrack.settingup.settingup import SettingUpCaller
    from covertrack.segmentation.call_segmentation import SegmentationCaller
    from covertrack.utils.seg_utils import find_label_boundaries
    from covertrack.segmentation import segmentation_operations
except:
    from settingup.settingup import SettingUpCaller
    from segmentation.call_segmentation import SegmentationCaller
    from utils.seg_utils import find_label_boundaries
    from segmentation import segmentation_operations

from ipywidgets import *
from traitlets import link
from IPython.display import display
import matplotlib.pyplot as plt
from traitlets import link
from copy import deepcopy
from PIL import Image
from skimage.exposure import equalize_hist, histogram, equalize_adapthist
from skimage.morphology import dilation
import numpy as np
import matplotlib.colors as mpcol
from scipy.ndimage import imread
import inspect
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets
from functools import partial
import imp
from os.path import dirname, join, abspath, basename, exists


'''
Using jupyter notebook so that you can play with parameters on a server
'''


Caller = SegmentationCaller


class text_storage(object):
    pass


class SegmentOptimizer(object):
    def __init__(self, input_path, segment_func, frame=0):
        argfile = imp.load_source('inputArgs', input_path)
        outputdir = join(argfile.output_parent_dir, basename(argfile.input_parent_dir))

        outputdir = SettingUpCaller(outputdir, input_path, None).run()
        sc = ModifiedSegmentCaller(outputdir)
        sc.argdict['segment_args'][0]['name'] = segment_func
        sc.frame = frame
        sc.set_obj_ch(sc.argdict['segment_args'])
        # load image
        sc.iter_channels()
        self.img = sc.store_img.copy()

        # make a function
        func_args = sc.argdict['segment_args'][-1]
        func_args = func_args.copy()
        func_name = func_args.pop('name')
        func = getattr(segmentation_operations, func_name)
        self.func = func
        ins = inspect.getargspec(func)
        param_lists = [i for i in ins.args if i.isupper()]
        dic = {}
        for key, values in zip(param_lists, ins.defaults):
            dic[key] = widgets.FloatText(values)
        self.params = dic
        self.run()

    def run(self):
        w1 = widgets.interactive(self.vis_child, **self.params)
        display(w1)

    def vis_child(self, **params):
        pfunc = partial(self.func, img=self.img.copy(), holder=None)
        label = pfunc(**params)
        plot_segment(self.img, label)


class ModifiedSegmentCaller(SegmentationCaller):
    def iter_frames(self):
        imgpath = self.pathset[self.frame]
        self.store_img = imread(imgpath)



def create_eval_args(param_lists):
    '''
    e.g.
    >>> param_lists = ['PAR1', 'PAR2]
    'widgets.interactive(func,PAR1=storage.PAR1,PAR2=storage.PAR2)'
    '''
    list_param_lists = [''.join([i, '=storage.', i]) for i in param_lists]
    eval_args = ','.join(list_param_lists)
    eval_args = 'widgets.interactive(func,' + eval_args + ')'
    return eval_args


def alg_construction(argset, segment_func, segment_param=[]):
    '''return the first alg. will be modified in the future'''
    algs_constructor = OC(argset, segmentation_algs, segment_func, segment_param)
    alg = algs_constructor.construct()[0]
    return alg


def make_textbox(alg, param_lists):
    att_and_values = [i for i in alg.__dict__.iteritems() if i[0] in param_lists]
    storage = text_storage()
    for i, j in att_and_values:
        setattr(storage, i, FloatText(j))
    return storage


def interactive_segment(input_path, segment_func, FRAME=0):
    seg, argset = prepare_seg(input_path, FRAME)
    alg = alg_construction(argset, segment_func)
    param_lists = [i for i in dir(alg) if i.isupper()]
    storage = make_textbox(alg, param_lists)  # used in eval
    eval_args = create_eval_args(param_lists)
    func = vis_child(seg.image, seg.params, argset, segmentation_algs, segment_func)  # used in eval
    w1 = eval(eval_args)
    display(w1)


def vis_child(image, params, argset, segmentation_algs, segment_func):
    def wrapper(**segment_param):
        alg = alg_construction(argset, segment_func, [segment_param, ])
        alg.set_input(image, params)
        alg.implement()
        output_image = alg.return_output()
        plot_segment(output_image)
    return wrapper


def prepare_seg(input_path, FRAME):
    setup = SettingUp(None, input_path)
    argset = setup.implement()
    seg = Caller(argset)
    seg.load_operation_constructor()
    seg.frame = FRAME
    seg.imgpath = seg.argset.channeldict[seg.channels[0]][FRAME]
    seg.load_image()
    return seg, argset


def call_vis(input_path, segment_func):
    seg, argset = prepare_seg(input_path)
    vis(seg, segment_func, argset)


def plot_segment(img, label):
    cim=image_comp()
    # cim.implement(output_image.img,output_image.label,cell_channel=1,label_channel=0,normalization=2,trans_value=0.5,label_type=1,bright=2)
    cim.implement(img,label,cell_channel=1,label_channel=0,normalization=2,trans_value=0.5,label_type=1,bright=2)



class image_comp(object):

    def __init__(self):
        pass

    def implement(self,user_cell, user_label, cell_channel=1, label_channel=2, label_type=1, dilation=1, \
                  view_option=1, normalization=1, normalization_value=0.8, bright=1, trans_value=1, user_cmap='rand', view_size=10):
        #cell and label images are input as paths?
        #cell and label_channel: 0-red, 1-green, 2-blue
        #label_type: 1-whole nuclear label, 2-borders only
        #dilation: 0-no dilation, 1-use grey-scale dilation
        #view_option: 1-all, 2-side by side only, 3-overlay only
        #normalization: 0-no normalization, 1-normalize label only, 2-normalize both
        #normalization_value: value in (0,1) to normalize label image to. Higher value is darker labels
        #trans= value in (0,1) is the level of transparency for non-black pixels
        #bright=scalar value that increases the intensity of each pixel when equalizing
        #view_size: size of each individual figure
#         user_cell=imread(cell_image)
#         user_label=imread(label_image)
        user_display_label=deepcopy(user_label)

        if label_type == 2:
            user_label=find_label_boundaries(user_label)

        user_cell=self.get_channel(user_cell,cell_channel)
        user_label=self.get_channel(user_label,label_channel)

        if normalization == 1:
            user_label=self.normalize(user_label,normalization_value)
        elif normalization == 2:
            user_label=self.normalize(user_label,normalization_value)
            user_cell=self.equalize(user_cell,bright)

        if dilation == 1:
            user_label=self.dilate(user_label)

        user_label=self.make_transparent(user_label,trans_value)

        if view_option == 1:
            self.plot_images(user_cell,user_display_label,cmap2=user_cmap,imsize=view_size)
            user_overlay=self.overlay_images(user_cell,user_label)
            self.plot_images(user_overlay,imsize=view_size)
        elif view_option == 2:
            self.plot_images(user_cell,user_display_label,cmap2=user_cmap,imsize=view_size)
        elif view_option == 3:
            user_overlay=self.overlay_images(user_cell,user_label)
            self.plot_images(user_overlay,imsize=view_size)

    def get_channel(self, user_image, user_channel):
        #Input image as PNG, return RGB
        #User channel specifies RGB (0-2)


        if len(user_image.shape)==2:
            user_image=np.dstack((user_image, user_image, user_image))
        elif len(user_image.shape)==3:
            pass

        user_image=user_image.astype('float')
        image_rgb=user_image*255/user_image.max()
        image_channel=np.zeros(shape=image_rgb.shape)
        image_channel[:,:,user_channel]=deepcopy(image_rgb[:,:,user_channel])
        image_channel=image_channel.astype('uint8')
        image_channel=Image.fromarray(image_channel,'RGB')

        return image_channel

    def make_transparent(self, user_image,alph):
        #Input image as PNG, return RGBA with transparancies over black pixels
        #Only works for images with one channel (R, G, or B)

        #can this be done without a for loop?
        for i in range(0,3):
            if np.sum(np.array(user_image)[:,:,i])>0:
                user_channel=i

        user_image=user_image.convert('RGBA')
        user_image_data=user_image.getdata()

        trans=[]; alph=int(alph*255) #why is int necessary?
        for item in user_image_data:
            if item[user_channel]!=0:
                trans.append((item[0],item[1],item[2],alph))
            else:
                trans.append((0,0,0,0))

        user_image.putdata(trans)

        return user_image

    def normalize(self,user_image,norm):
        #Takes an RGB image and normalizes to the specified value (0-1)

        user_image=user_image.convert('RGBA')
        user_image_data=user_image.getdata()

        normal=[]; norm=int(norm*255)
        for item in user_image_data:
            if item[0]!=0:
                normal.append((norm,item[1],item[2],item[3]))
            elif item[1]!=0:
                normal.append((item[0],norm,item[2],item[3]))
            elif item[2]!=0:
                normal.append((item[0],item[1],norm,item[3]))
            else:
                normal.append((0,0,0,0))

        user_image.putdata(normal)

        return user_image

    def overlay_images(self,image1,image2):
        image1.paste(image2,(0,0),image2)
        return(image1)

    def equalize(self,user_image,bright=1):

        image_array=equalize_adapthist(np.array(user_image))
        image_array=image_array*255*bright
        image_array=image_array.astype('uint8')

        if image_array.shape[2]==3:
            image_eq=Image.fromarray(image_array,'RGB')
        elif image.array.shape[2]==4:
            image_eq=Image.fromarray(image_array,'RGBA')

        return image_eq

    def dilate(self,user_image):
        #inputs RGB or RGBA image and outputs with grey scale dilation

        image_array=np.array(user_image)
        image_dilated=dilation(image_array)
        image_dilated=image_dilated.astype('uint8')

        if image_array.shape[2]==3:
            image_out=Image.fromarray(image_dilated,'RGB')
        elif image_array.shape[2]==4:
            image_out=Image.fromarray(image_dilated,'RGBA')

        return image_out

    def plot_images(self,image1,image2=None,cmap1=None,cmap2=None,back=211, imsize=None):

        if imsize==None:
            imsize=10

        back=back/255.
        cmaprand =np.random.rand (256,3)
        cmaprand[0,:]=[back,back,back]
        cmaprand=mpcol.ListedColormap(cmaprand)

        if cmap1=='rand':
            cmap1=cmaprand
        elif cmap2=='rand':
            cmap2=cmaprand

        if image2==None:
            fig=plt.figure(figsize=(imsize,imsize))
            plt.axis('off')
            imgplot=plt.imshow(image1,cmap=None)
        else:
            fig=plt.figure(figsize=(2*imsize,2*imsize))
            a=fig.add_subplot(1,2,1)
            plt.axis('off')
            imgplot=plt.imshow(image1,cmap=cmap1)
            a=fig.add_subplot(1,2,2)
            plt.axis('off')
            imgplot=plt.imshow(image2,cmap=cmap2)

if __name__ == "__main__":

    so = SegmentOptimizer('/Users/kudo/ktr_protocol/covertrackdev/input_file/input_tests2.py', 'constant_lap_edge')
    so.run()
