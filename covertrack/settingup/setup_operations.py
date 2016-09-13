import glob
from os.path import join
import os


def retrieve_files(argdict, imgdir, holder, channels):
    chdict = {}
    for ch in channels:
        chdict[ch] = sorted([join(imgdir, i) for i in os.listdir(imgdir) if ch in i])
    argdict['channeldict'] = chdict
    argdict['channels'] = channels
    return argdict


def retrieve_files_glob(argdict, imgdir, holder, channels, patterns):
    chdict = {}
    for ch, pattern in zip(channels, patterns):
        chdict[ch] = glob.glob(join(imgdir, pattern))
    argdict['channeldict'] = chdict
    argdict['channels'] = channels
    return argdict
