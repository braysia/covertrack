'''
You need to add covertrack directory to PYTHONPATH
e.g. export PYTHONPATH="$PYTHONPATH:$PI_SCRATCH/kudo/covertrack"
'''

import os
import sys
from os.path import join, basename, exists
sys.path.append(join(os.path.dirname(os.path.abspath(__file__)), 'covertrack'))
from covertrack.caller import CovertrackArgs
from covertrack.utils.file_handling import ImgdirsFinder
from fireworks import FireTaskBase, explicit_serialize
from fireworks import Firework, LaunchPad, Workflow
import yaml
# from utils.file_handling import find_imgdirs
import imp
from covertrack.caller import PROCESSES
import argparse
from argparse import Namespace

@explicit_serialize
class clustercovertrack(FireTaskBase):
        _fw_name = "clustercovertrack"
        required_params = ["input_args_path", "imgdir", "args"]
        def run_task(self, fw_spec):
            print "Running clustercovertrack with input {0} and {1}".format(self["input_args_path"], self["imgdir"])
            parallel_call_analysis(self["input_args_path"], self["imgdir"], self["args"])


def initiate_cluster(ia_path, args):
    # check how many image folders are there
    imgdirs = read_imgdirs_from_parentdir(ia_path)
    if args.skip:
        imgdirs = ignore_if_df_existed(imgdirs, ia_path)
    lpad = LaunchPad(**yaml.load(open("my_launchpad.yaml")))
    wf_fws = []
    for iv in imgdirs:
        # start loop over input val
        fw_name = "clustercovertrack"
        fw = Firework(
                clustercovertrack(input_args_path=ia_path, imgdir=iv, args=args),
                name = fw_name,
                spec = {"_queueadapter": {"job_name": fw_name, "walltime": "47:00:00"}},
        )
        wf_fws.append(fw)
    # end loop over input values
    workflow = Workflow(wf_fws, links_dict={})
    lpad.add_wf(workflow)


def read_imgdirs_from_parentdir(ia_path):
    ia = imp.load_source('inputArgs', ia_path)
    imgdirs = ImgdirsFinder(ia.input_parent_dir).find_dirs()
    return imgdirs


def ignore_if_df_existed(imgdirs, ia_path):
    ia = imp.load_source('inputArgs', ia_path)
    output_parent_dir = ia.output_parent_dir
    new_imgdirs = []
    for imgdir in imgdirs:
        if not exists(join(output_parent_dir, basename(imgdir), 'df.npz')):
            new_imgdirs.append(imgdir)
    return new_imgdirs


class parallel_call_analysis():
    def __init__(self, ia_path, imgdir, args):
        args = eval(args)
        CovertrackArgs(ia_path, imgdir, args).run()


def call_cluster(ia_path, args):
    # os.system('lpad reset')
    initiate_cluster(ia_path, args)
    # launch_cmd = 'qlaunch -r rapidfire -m '+str(int(sys.argv[1])+1)+' --nlaunches infinite'
    # launch_cmd = 'qlaunch -r rapidfire -m '+20+' --nlaunches infinite'
    # os.system(launch_cmd)


if __name__ == "__main__":
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
    parser.add_argument("-s", "--skip", help="skip if df.npz is already created",
                        action="store_true")
    parser.add_argument("input", nargs="*", help="input argument file path")
    args = parser.parse_args()
    # If nothing is specified, then set it to True
    if not any([getattr(args, i) for i in PROCESSES]):
        [setattr(args, i, True) for i in PROCESSES]
    call_cluster(args.input[0], args)
