from os.path import join, dirname, abspath
import os

ROOT_FOLDER = dirname(dirname(abspath(__file__)))
input_parent_dir = join(ROOT_FOLDER, 'data', 'testimages2')
output_parent_dir = join(ROOT_FOLDER, 'tests', 'output')
# Keep None to analyze all
first_frame = None
last_frame = None

channels = ['Far-red', 'YFP']
objects = ['nuclei', 'cytoplasm']  # first object will be used for tracking

setup_args = ((dict(name='retrieve_files', channels=['Far-red', 'YFP'])), )

preprocess_args = (dict(name='background_subtraction_wavelet'), )

segment_args = (dict(name='constant_lap_edge', SHRINK=1, NUCRAD=9, THRES=120, REGWSHED=10),)

# Tracking
_param_runlap = dict(name='run_lap', DISPLACEMENT=50, MASSTHRES=0.2)
_param_tn = dict(name='track_neck_cut', DISPLACEMENT=50, MASSTHRES=0.2, EDGELEN=7)
_paramwd = dict(name='watershed_distance', DISPLACEMENT=50, MASSTHRES=0.2, ERODI=6)
track_args = (_param_runlap, _param_tn, _paramwd)

# Postprocessing
postprocess_args = (dict(name='detect_division', DISPLACEMENT=20, DIVISIONMASSERR=0.1),
                    dict(name='gap_closing'),
                    dict(name='cut_short_traces', minframe=2))
# Subdetection
subdetect_args = (dict(name='ring_dilation', object_name='cytoplasm',
                       seed_obj='nuclei', MARGIN=1, RINGWIDTH=5), )
