"""
Please make sure to assign environment variables $COVERTRACK and $PYTHONPATH.

Then run following commands:
    python call_fireworks.py {ABSOLUTE_PATH to covertrack/input_file/input_tests_fireworks.py}
    qlaunch -r rapidfire -m 3 --nlaunches infinite


"""


import os
from os.path import join, dirname, abspath

input_parent_dir = join(os.environ['COVERTRACK'], 'data', 'test_fireworks')
output_parent_dir = join(os.environ['COVERTRACK'], 'tests', 'output', 'fireworks')

first_frame = None
last_frame = None


setup_args = ((dict(name='retrieve_files', channels=['CFP', 'YFP'])), )

preprocess_args = (dict(name='background_subtraction_prcblock'), )

segment_args = (dict(name='example_thres', object_name='nuclei', THRES=1500),)

track_args = (dict(name='nearest_neighbor', DISPLACEMENT=15, MASSTHRES=0.2), )

postprocess_args = (dict(name='cut_short_traces', minframe=2), )
subdetect_args = (dict(name='ring_dilation', object_name='cytoplasm',
                       seed_obj='nuclei', MARGIN=1, RINGWIDTH=5), )
