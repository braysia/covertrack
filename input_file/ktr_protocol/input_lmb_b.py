import os
from os.path import abspath, dirname, join

ROOTDIR = dirname(dirname(dirname(abspath(__file__))))

input_parent_dir = join(ROOTDIR, 'data', 'ktr_images', 'LMB', 'Pos005')
output_parent_dir = join(ROOTDIR, 'output', 'LMB')

# Keep None to analyze all
first_frame = None
last_frame = None

objects = ['nuclei', 'cytoplasm']  # first object will be used for tracking

setup_args = ((dict(name='retrieve_files', channels=['DAPI', 'YFP'])), )

# Preprocessing for tracking
preprocess_args = (dict(name='hist_matching', ch='DAPI'),
                   dict(name='n4_illum_correction_downsample'),
                   dict(name='background_subtraction_wavelet_hazen', ch='DAPI', THRES=1500),
                   dict(name='smooth_curvature_anisotropic', ch='DAPI'))

# Segmentation
segment_args = (dict(name='lapgauss_adaptive', RATIO=1.2, SIGMA=2.5, REGWSHED=8),)

# Tracking
_param_runlap = dict(name='run_lap', DISPLACEMENT=50, MASSTHRES=0.3)
_param_tn = dict(name='track_neck_cut', DISPLACEMENT=50, MASSTHRES=0.2, EDGELEN=7)
_paramwd = dict(name='watershed_distance', DISPLACEMENT=50, MASSTHRES=0.2, ERODI=6)
track_args = (_param_runlap, _param_tn, _paramwd)

# Postprocessing
postprocess_args = (dict(name='gap_closing'),
                    dict(name='cut_short_traces', minframe=54))
# Subdetection
subdetect_args = (dict(name='ring_dilation_above_offset_buffer', object_name='cytoplasm',
                       seed_obj='nuclei', ch_img='YFP', RINGWIDTH=5, BUFFER=4, OFFSET=300, FILSIZE=25), )
