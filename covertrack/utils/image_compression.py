import math
from PIL import Image
import subprocess
import os
import json
from os.path import join

def scoreatpercentile(N, per, key=lambda x: x):
    """
    Find the percentile of a list of values.
    @parameter N - is a list of values.
    @parameter per - a float value from 0 to 100.
    @parameter key - optional key function to compute value from each element of N.
    @return - the percentile of the values
    """
    frac = per / 100.0
    N.sort()
    k = (len(N)-1) * frac
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return key(N[int(k)])
    d0 = key(N[int(f)]) * (c-k)
    d1 = key(N[int(c)]) * (k-f)
    return d0+d1


def getLoHiForImages(pathset, percentiles):
    loHiAll = [[], []]
    for filename in pathset:
        img = Image.open(filename)
        pixList = list(img.getdata())

        for ii in range(2):
            loHiAll[ii].append(scoreatpercentile(pixList, percentiles[ii]))

    loHiLevel = []
    loHiLevel.append(max(loHiAll[0]))
    loHiLevel.append(min(loHiAll[1]))

    if img.mode == 'I':
        depth = 16
    else:
        depth = 8
    loHiPercent = [level / (2.0**depth-1) * 100 for level in loHiLevel]

    return loHiPercent


def compressChannelImages(pathset, outputDir):
    # if compressing to 8-bit png, make sure to specify -depth 8
    percentiles = [1, 99.99] # from 0 to 100 (used for rescaling compressed images)
    sepChar = '_'
    outputExt = '.jpg'
    quality = 95
    outputDir = os.path.join(outputDir, 'channels')
    loHi = getLoHiForImages(pathset, percentiles)
    try:
        for path in pathset:
            inputDir, filename = os.path.split(path)
            inputPath = os.path.join(inputDir, filename)
            outputPath = os.path.join(outputDir, os.path.splitext(filename)[0]+outputExt)
            subprocess.call(['convert', inputPath, '-level', str(loHi[0])+'%,'+str(loHi[1])+'%','-quality', str(quality), outputPath])
    except SyntaxError:
        print 'to convert images you may need to install imagemagick'

def compress_channel_image(outputdir):
    with open(join(outputdir, 'setting.json')) as file:
        argdict = json.load(file)
    for pathset in argdict['channeldict'].itervalues():
        compressChannelImages(pathset, outputdir)
