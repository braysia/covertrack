from os.path import join, dirname
import os

input_parent_dir = join(dirname(dirname(os.path.abspath(__file__)))	, 'covertrackdev', 'tests', 'testimages2')
output_parent_dir = join(dirname(dirname(os.path.abspath(__file__))),	'covertrackdev', 'tests', 'output')

input_parent_dir = '/Users/kudo/temp/cell_seg_data/raw/Pos58'
output_parent_dir = '/Users/kudo/temp/cell_seg_data/'

# Keep None to analyze all
first_frame = None
last_frame = None

channels = ['Far-red', 'YFP', 'TRITC', 'CFP']
objects = ['nuclei', 'cytoplasm']  # first object will be used for tracking


preprocess_args = (dict(ch='CFP', name='n4_illum_correction'),
                   dict(name='background_subtraction_wavelet_hazen', THRES=1500),
                   dict(ch='CFP', name='smooth_curvature_anisotropic'))

# Segmentation
segment_args = (dict(name='constant_lap_edge', DEBRISAREA=50, MAXSIZE=1000,
                     SHRINK=1, NUCRAD=9, THRES=120, REGWSHED=8),)
# segment_args = (dict(name='adaptivethreshwithglobal_neckcut'), )

# Tracking
_param_jitter = dict(name='jitter_correction_label')
_param_runlap = dict(name='run_lap', DISPLACEMENT=30, MASSTHRES=0.3)
_paramnn = (dict(name='nearest_neighbor', DISPLACEMENT=20, MASSTHRES=0.2))
# _paramwd = dict(name='track_neck_cut', DISPLACEMENT=50, MASSTHRES=0.2, EDGELEN=5)
# _param_nn2 = dict(name='track_neck_cut', DISPLACEMENT=50, MASSTHRES=0.3, EDGELEN=3)
_paramwd2 = dict(name='watershed_distance', DISPLACEMENT=50, MASSTHRES=0.2, ERODI=7)
_paramtc = dict(name='track_neck_cut', DISPLACEMENT=15, MASSTHRES=0.3, EDGELEN=5)
_paramtc2 = dict(name='track_neck_cut', DISPLACEMENT=20, MASSTHRES=0.3, EDGELEN=3)

# track_args = (_param_jitter, _param_runlap, _paramwd, _param_nn2, _paramwd2)
track_args = (_param_runlap, _paramnn, _paramtc, _paramtc2)

# Postprocessing
postprocess_args = (dict(name='detect_division', DISPLACEMENT=50, maxgap=4, DIVISIONMASSERR=0.15),
                    dict(name='gap_closing', DISPLACEMENT=100, MASSTHRES=0.5, maxgap=4),
                    dict(name='cut_short_traces', minframe=20))
# Subdetection
"""
ch_img can be str of list of str. If it's a list, self.img will be 3D numpy array.
"""
subdetect_args = (dict(name='ring_dilation', object_name='cytoplasm', seed_obj='nuclei', ch_img=['Far-red', 'YFP'], MARGIN=5, RINGWIDTH=8), )
subdetect_args = (dict(name='segment_bacteria', object_name='cell', seed_obj='nuclei', ch_img='CFP'), )
