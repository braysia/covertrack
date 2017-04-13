import _import
from os.path import join
from settingup.settingup import SettingUpCaller
from preprocess.call_preprocessing import PreprocessCaller
from segmentation.call_segmentation import SegmentationCaller
from tracking.call_tracking import TrackingCaller
from postprocessing.call_postprocess import PostprocessCaller
from subdetection.subdetection import SubDetection
from apply_objects.apply_objects import ApplyObjects
from os.path import dirname, join, abspath, basename, exists
import os
from utils.image_compression import compress_channel_image
import multiprocessing
from joblib import Parallel, delayed
import imp
import argparse
from argparse import Namespace
from logging import getLogger, StreamHandler, FileHandler, DEBUG, WARNING, INFO
import shutil
import numpy as np
import re
import tempfile, shutil


PROCESSES = ('setup', 'preprocess', 'segment', 'track', 'postprocess',
             'subdetect', 'apply', 'compress')
np.random.seed(0)


class Covertrack(object):
    def __init__(self, ia_path, imgdir=None):
        self.ia_path = ia_path
        self.imgdir = imgdir

    def run(self):
        argfile = imp.load_source('inputArgs', self.ia_path)
        outputdir = join(argfile.output_parent_dir, basename(argfile.input_parent_dir))

        outputdir = SettingUpCaller(self.ia_path, self.imgdir).run()
        PreprocessCaller(outputdir).run()
        SegmentationCaller(outputdir).run()
        TrackingCaller(outputdir).run()
        compress_channel_image(outputdir)
        PostprocessCaller(outputdir).run()
        SubDetection(outputdir).run()
        ApplyObjects(outputdir).run()


class CovertrackArgs(Covertrack):
    def __init__(self, ia_path, imgdir=None, args=None):
        self.ia_path = ia_path
        self.imgdir = imgdir
        self.args = args

    def run(self):
        outputdir = self.set_output()
        if self.args.clean:
            if exists(outputdir):
                shutil.rmtree(outputdir)
        self.establish_logger(outputdir, self.args.quiet, self.args.verbose)
        if self.args.setup:
            SettingUpCaller(outputdir, self.ia_path, self.imgdir).run()
        if self.args.preprocess:
            PreprocessCaller(outputdir).run()
        if self.args.segment:
            SegmentationCaller(outputdir).run()
        if self.args.track:
            TrackingCaller(outputdir).run()
        if self.args.compress:
            compress_channel_image(outputdir)
        if self.args.postprocess:
            PostprocessCaller(outputdir).run()
        if self.args.subdetect:
            SubDetection(outputdir).run()
        if self.args.apply:
            ApplyObjects(outputdir).run()
        if self.args.delete:
            self._attempt_delete(outputdir)

    def set_output(self):
        argfile = imp.load_source('inputArgs', self.ia_path)
        outputdir = join(argfile.output_parent_dir, basename(argfile.input_parent_dir))
        if self.imgdir:
            outputdir = join(argfile.output_parent_dir, basename(self.imgdir))
        return outputdir

    def _attempt_delete(self, outputdir):
        for f in ('cleaned', 'segmented', 'tracked', 'storage', 'processed'):
            shutil.rmtree(join(outputdir, f))
        for file in ('df.csv', 'ini_df.csv', 'pathset.npy'):
            os.remove(join(outputdir, file))

    def establish_logger(self, outputdir, _q=False, _v=False):
        '''prepare logging setting
        '''
        if not exists(outputdir):
            os.makedirs(outputdir)
        logger = getLogger('covertrack')
        fh = FileHandler(join(outputdir, 'log.txt'))
        handler = StreamHandler()
        handler.setLevel(DEBUG)
        if _q:
            logger.setLevel(WARNING)
        elif _v:
            logger.setLevel(DEBUG)  # pick one
        else:
            logger.setLevel(INFO)
        logger.addHandler(handler)
        logger.addHandler(fh)


class MetamorphArgs(CovertrackArgs):
    def __init__(self, ia_path, imgdir=None, args=None, site=1):
        self.ia_path = ia_path
        self.imgdir = imgdir
        self.args = args
        self.site = site

    def set_output(self):
        argfile = imp.load_source('inputArgs', self.ia_path)
        outputdir = join(argfile.output_parent_dir, 's{0}'.format(self.site))
        self.imgdir = argfile.input_parent_dir
        self.set_setup()
        return outputdir

    def set_setup(self):
        argfile = imp.load_source('inputArgs', self.ia_path)
        pattern = "_w[0-9]*(?P<ch>.*)_s(?P<site>[0-9]*)_t(?P<ts>[0-9]*).(?P<format>.*)"
        store = []
        for f in os.listdir(argfile.input_parent_dir):
            store.append(re.search(pattern, f))
        f_site = []
        for i in store:
            try:
                if int(i.groupdict()['site']) == self.site:
                    f_site.append(i)
            except:
                pass
        channels = set([f.groupdict()['ch'] for f in f_site])
        channels = list(channels)
        temp_path = join(tempfile.mkdtemp(), basename(self.ia_path))
        shutil.copyfile(self.ia_path, temp_path)

        chp = []
        for i in channels:
            chp.append('*{0}_s{1}_*'.format(i, self.site))
        texts = "setup_args = ((dict(name='retrieve_files_re_ts', channels={0}, chpatterns={1}, re_ts = 't(?P<ts>[0-9]*).TIF')), )".format(channels, chp)
        with open(temp_path, 'a') as file:
            file.writelines(texts)
        self.ia_path = temp_path


def single_call(input_path, imgdir=None, args=None, site=1):
    MetamorphArgs(input_path, imgdir, args, site).run()


def call_help_ops():
    """
    Call help(*_operations) in a loop.
    """
    from settingup import setup_operations
    from preprocess import preprocess_operations
    from segmentation import segmentation_operations
    from tracking import tracking_operations
    from postprocessing import postprocessing_operations
    from subdetection import subdetection_operations
    ops = [i for i in dir() if i.endswith('operations')]
    for op in ops:
        print help(eval(op))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-1", "--setup", help="run settingup",
                        action="store_true")
    parser.add_argument("-2", "--preprocess", help="run preprocessing",
                        action="store_true")
    parser.add_argument("-3", "--segment", help="run segmentation",
                        action="store_true")
    parser.add_argument("-4", "--track", help="run tracking",
                        action="store_true")
    parser.add_argument("-5", "--postprocess", help="run postprocessing",
                        action="store_true")
    parser.add_argument("-6", "--subdetect", help="run subdetection",
                        action="store_true")
    parser.add_argument("-7", "--apply", help="run applyobjects",
                        action="store_true")
    parser.add_argument("-8", "--compress", help="run image compression",
                        action="store_true")
    parser.add_argument("-q", "--quiet", help="set logging level to WARNING",
                        action="store_true")
    parser.add_argument("-v", "--verbose", help="set logging level to DEBUG",
                        action="store_true")
    parser.add_argument("-d", "--delete", help="delete extra files to save space",
                        action="store_true")
    parser.add_argument("-c", "--clean", help="delete analyzed files if existed",
                        action="store_true")
    parser.add_argument("-n", "--cores", help="number of cores for multiprocessing",
                        type=int)
    parser.add_argument("-l", "--list", help="list all operations by calling help",
                        action="store_true")
    parser.add_argument("input", nargs="*", help="input argument file path")
    args = parser.parse_args()

    if args.list:
        call_help_ops()
    if not any([getattr(args, i) for i in PROCESSES]):
        [setattr(args, i, True) for i in PROCESSES]

    argfile = imp.load_source('inputArgs', args.input[0])
    if not hasattr(argfile, "SITES"):
        print "include SITES (List(int)) in your input file."
    if len(argfile.SITES) == 1:
        MetamorphArgs(args.input[0], None, args).run()
    if len(argfile.SITES) > 1:
        num_cores = args.cores
        print str(num_cores) + ' started parallel'
        Parallel(n_jobs=num_cores)(delayed(single_call)(args.input[0], None, args, i) for i in argfile.SITES)

if __name__ == "__main__":
    main()
