import glob
from os.path import join
import os


def retrieve_files(argdict, imgdir, holder, channels):
    chdict = {}
    for ch in channels:
        pathset = [join(imgdir, i) for i in os.listdir(imgdir) if ch in i]
        chdict[ch] = sorted(pathset)[argdict['first_frame']:argdict['last_frame']]
    argdict['channeldict'] = chdict
    argdict['channels'] = channels
    return argdict


def retrieve_files_glob(argdict, imgdir, holder, channels, patterns):
    chdict = {}
    for ch, pattern in zip(channels, patterns):
        pathset = glob.glob(join(imgdir, pattern))
        chdict[ch] = pathset[argdict['first_frame']:argdict['last_frame']]
    argdict['channeldict'] = chdict
    argdict['channels'] = channels
    return argdict
