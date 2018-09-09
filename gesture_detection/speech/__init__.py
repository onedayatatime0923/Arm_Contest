#coding=utf-8

import os
import multiprocessing as mp

assert mp

class Speech:
    def __init__(self):
        self.translate = {
                'I': '我',
                'name': '名字',
                'is': '是',
                'dog': '小狗'}
    def _speech(self, data):
        os.system("google_speech -l zh-TW {}".format(self.translate[data]))
    def __call__(self, data):
        self._speech(data)
        '''
        p = mp.Process(target=self._speech, args=(data,))
        p.start()
        '''
