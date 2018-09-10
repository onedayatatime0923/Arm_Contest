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
                'dog': '小狗',
                'big': '大',
                'home': '家',
                'good': '好',
                'we': '我們',
                'today': '今天',
                'come': '來',
                'participate': '參加',
                'make': '做出',
                'help': '幫助',
                'deaf': '聾',
                'dumb': '啞',
                'human': '人士',
                'of': '的',
                'gloves': '手套',
                'rat': '老鼠'}
    def _speech(self, data):
        if data in self.translate:
            os.system("google_speech -l zh-TW {}".format(self.translate[data]))
        else:
            os.system("google_speech -l en {}".format(data))
    def __call__(self, data):
        self._speech(data)
