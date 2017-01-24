import glob
from os.path import join
import os


def retrieve_files(argdict, imgdir, holder, channels):
    """
    Use this function if image files contain a channel name (e.g. img_*_CFP*.png).
    channels (list): a list of channel name.
    """
    chdict = {}
    for ch in channels:
        pathset = [join(imgdir, i) for i in os.listdir(imgdir) if ch in i]
        chdict[ch] = sorted(pathset)[argdict['first_frame']:argdict['last_frame']]
    argdict['channeldict'] = chdict
    argdict['channels'] = channels
    return argdict


def retrieve_files_glob(argdict, imgdir, holder, channels, patterns):
    """
    patterns (list): a list of regular expression pattern used to extract files using glob.
    """
    chdict = {}
    for ch, pattern in zip(channels, patterns):
        pathset = glob.glob(join(imgdir, pattern))
        chdict[ch] = sorted(pathset)[argdict['first_frame']:argdict['last_frame']]
    argdict['channeldict'] = chdict
    argdict['channels'] = channels
    return argdict
