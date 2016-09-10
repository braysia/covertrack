import numpy as np
import simplejson
import re
from datetime import datetime
from logging import getLogger, StreamHandler, FileHandler, DEBUG
from os.path import join, basename, dirname, exists
import os


class NikonMetadata():
    logger = getLogger('covertrack.settingup')
    magnification = 0
    binning = 0
    img_shape = []
    time = []

    def __init__(self, argset, imgdir):
        self.inputdir = imgdir
        self.argset = argset
        self.retrieve_json()


    def retrieve_json(self):
        try:
            print self.inputdir
            f = open(join(self.inputdir, 'metadata.txt')).read()
            try:
                self.js = simplejson.loads(f)
            except:
                self.js = simplejson.loads(f+'}')
        except:
            self.logger.info('No metadata exists...')

    def _extract_magnification(self):
        try:
            objLabel = self.js[self.js.keys()[0]]['Objective Turret-Label']
            self.magnification = int(re.search('[0-9]*-(?P<mag>[0-9]*)X', objLabel).group('mag'))
            return self.magnification
        except:
            self.logger.debug('magnification not extracted')

    def _extract_img_shape(self):
        try:
            img_height = self.js[self.js.keys()[0]]['Height']
            img_width = self.js[self.js.keys()[0]]['Width']
            self.img_shape = (img_height, img_width)
            return self.img_shape
        except:
            self.logger.warn('img shape not extracted. The first image will be used. Otherwise please specify img_shape in inputArgs.')

    def _extract_time(self):
        try:
            keynames = [i for i in self.js.keys() if i != 'Summary']
            frame_num_list = [int(re.search('FrameKey-(?P<f>[0-9]*)', i).group('f')) for i in keynames]
            frames = np.unique(frame_num_list)
            keynames_to_read = ['FrameKey-'+str(i)+'-0-0' for i in frames]
            timeList = [self.js[i]['Time'] for i in keynames_to_read]
            times =[]
            for i in timeList:
                matched = re.search('(?P<year>[0-9]*)-(?P<month>[0-9]*)-(?P<day>[0-9]*) (?P<hours>[0-9]*):(?P<minutes>[0-9]*):(?P<seconds>[0-9]*)', i)
                tempDate = matched.groups()
                times.append(datetime(int(tempDate[0]),int(tempDate[1]),int(tempDate[2]),int(tempDate[3]),int(tempDate[4])))
            timediff = map(lambda x: x - times[0], times)
            hours = [i.days*24. + i.seconds/3600. for i in timediff]
            self.time = hours
            return self.time
        except:
            self.logger.debug('time not extracted')
            self.time = range(len(self.argset.channeldict.values()[0]))

    def _extract_binning(self):
        try:
            self.binning = int(self.js[self.js.keys()[0]]['Binning'][0])
            return self.binning
        except:
            self.logger.debug('binning not extracted')

    # def extract_xyposition(self):
    #     self.xPosition = self.js[self.js.keys()[0]]['XPositionUm']
    #     self.yPosition = self.js[self.js.keys()[0]]['YPositionUm']

    # def extract_wellposition(self):
    #     try:
    #         self.extract_xyposition()
    #         wellPos = self.closest_well(self.xPosition, self.yPosition)
    #     except:
    #         self.logger.debug('well position not extracted')
    #     return wellPos

    # @staticmethod
    # def closest_well(xPosition, yPosition):
    #     xydata = loadmat('flatimgs/wellposition96.mat')
    #     xWell = np.argmin(abs(xPosition - xydata['col']))
    #     yWell = np.argmin(abs(yPosition - xydata['row']))
    #     rowAlphabet = 'ABCDEFG'
    #     wellPosition = rowAlphabet[yWell] + str('%0.2d') % (xWell+1)
    #     return wellPosition
