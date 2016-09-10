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
from argparse import Namespace  # this is required for fireworks
from logging import getLogger, StreamHandler, FileHandler, DEBUG, WARNING, INFO


PROCESSES = ('setup', 'preprocess', 'segment', 'track', 'postprocess',
             'subdetect', 'apply', 'compress')


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
        self.establish_logger(outputdir, self.args.quiet, self.args.verbose)
        if self.args.setup:
            outputdir = SettingUpCaller(self.ia_path, self.imgdir).run()
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

    def set_output(self):
        argfile = imp.load_source('inputArgs', self.ia_path)
        outputdir = join(argfile.output_parent_dir, basename(argfile.input_parent_dir))
        if self.imgdir is not None:
            outputdir = join(outputdir, basename(self.imgdir))
        return outputdir

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


def single_call(input_path, imgdir=None, args=None):
    CovertrackArgs(input_path, imgdir, args).run()


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
    parser.add_argument("input", nargs="*", help="input argument file path")
    args = parser.parse_args()
    # If nothing is specified, then set it to True
    if not any([getattr(args, i) for i in PROCESSES]):
        [setattr(args, i, True) for i in PROCESSES]
    if len(args.input) == 1:
        CovertrackArgs(args.input[0], None, args).run()
    if len(args.input) > 1:
        num_cores = multiprocessing.cpu_count()
        num_cores = len(args.input) if len(args.input) < num_cores else num_cores
        print str(num_cores) + ' started parallel'
        Parallel(n_jobs=num_cores)(delayed(single_call)(i, None, args) for i in args.input)

if __name__ == "__main__":
    main()
