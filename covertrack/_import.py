import sys,os
from os.path import join

dirname = os.path.dirname(os.path.abspath(__file__))
# subdirs = [x[0] for x in os.walk(dirname)]
sys.path.append(dirname)
# sys.path.append(join(dirname, 'preprocess'))
# sys.path.append(join(dirname, 'apply_objects'))
# sys.path.append(join(dirname, 'postprocessing'))
# sys.path.append(join(dirname, 'segmentation'))
# sys.path.append(join(dirname, 'settingup'))
# sys.path.append(join(dirname, 'subdetection'))
# sys.path.append(join(dirname, 'tracking'))
# sys.path.append(join(dirname, 'utils'))
