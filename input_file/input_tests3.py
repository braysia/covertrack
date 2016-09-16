from os.path import join, dirname, abspath

ROOT_FOLDER = dirname(dirname(abspath(__file__)))
input_parent_dir = join(ROOT_FOLDER, 'data', 'testimages3')
output_parent_dir = join(ROOT_FOLDER, 'tests', 'output')
# Keep None to analyze all
first_frame = None
last_frame = None

setup_args = (dict(name='retrieve_files_glob', channels=['CFP', 'YFP'],
                   patterns=['*channel000*.png', '*channel001*.png']), )

preprocess_args = (dict(name='background_subtraction_prcblock'), )

segment_args = (dict(name='example_thres', THRES=1500),)

# Tracking
track_args = (dict(name='nearest_neighbor', DISPLACEMENT=15, MASSTHRES=0.2), )

# Postprocessing
postprocess_args = (dict(name='cut_short_traces', minframe=0), )
# Subdetection
subdetect_args = (dict(name='ring_dilation', object_name='cytoplasm',
                       seed_obj='nuclei', MARGIN=1, RINGWIDTH=5), )
