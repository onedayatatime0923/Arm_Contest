
import os.path
import numpy as np
from utils import Vocabulary

class Recorder():
    def __init__(self, opt):
        self.X = []
        self.Y = []
        self.opt = opt
        self.counter = 0
        self.vocabulary = Vocabulary()

    def __call__(self, data):
        self.label(data)
        # handle dumping data
        self.counter +=1
        if self.counter % self.opt.recordInterval == 0:
            self.dump()
            return True
        else:
            return False

    def label(self, data):
        self.X.append(data)
        self.Y.append(self.vocabulary.word2index[self.opt.action])

    def dump(self):
        index = 0
        path = os.path.join(self.opt.splitDir, '{}.npy'.format(index))
        while os.path.exists(path):
            path = os.path.join(self.opt.splitDir, '{}.npy'.format(index))
            index += 1
        print('saved data to {}...'.format(path))
        raw_input()
        np.save(path, np.array([self.X, self.Y]))
        self.X = []
        self.Y = []
