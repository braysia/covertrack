from os.path import join, dirname, abspath

ROOT_FOLDER = dirname(dirname(abspath(__file__)))
input_parent_dir = join(ROOT_FOLDER, 'data', 'testimages4')
output_parent_dir = join(ROOT_FOLDER, 'tests', 'output')
# Keep None to analyze all
first_frame = None
last_frame = None

setup_args = ((dict(name='retrieve_files', channels=['Far-red'])), )

preprocess_args = (dict(name='background_subtraction_wavelet'), )

segment_args = (dict(name='constant_lap_edge', SHRINK=1, NUCRAD=9, THRES=120, REGWSHED=10),)

# Tracking
_param_jitter = dict(name='jitter_correction_mutualinfo')
_param_runlap = dict(name='nearest_neighbor', DISPLACEMENT=20, MASSTHRES=0.2)
track_args = (_param_jitter, _param_runlap)

# Postprocessing
postprocess_args = (dict(name='detect_division', DISPLACEMENT=20, DIVISIONMASSERR=0.1),
                    dict(name='gap_closing'),
                    dict(name='cut_short_traces', minframe=2))
# Subdetection
subdetect_args = (dict(name='ring_dilation', object_name='cytoplasm',
                       seed_obj='nuclei', MARGIN=1, RINGWIDTH=5), )
