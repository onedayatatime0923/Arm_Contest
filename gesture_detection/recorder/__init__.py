
import os.path
import numpy as np

class Recorder():
    def __init__(self, opt):
        self.X = []
        self.Y = []
        self.opt = opt
        self.counter = 0

    def __call__(self, data):
        self.labeling(data)
        # handle dumping data
        self.counter +=1
        if self.counter % self.opt.recordInterval == 0:
            self.dump_data()

    def labeling(self, data):
        if self.opt.mode == 'stop':
            self.X.append(data)
            self.Y.append(0)
        elif self.opt.mode == 'move':
            self.X.append(data)
            self.Y.append(1)

    def dump_data(self):
        index = 0
        path = os.path.join(self.opt.dataDir, self.opt.split, '{}.npy'.format(index))
        while os.path.exists(path):
            path = os.path.join(self.opt.dataDir, self.opt.split, '{}.npy'.format(index))
            index += 1
        np.save(path, np.array([self.X, self.Y]))
        self.X = []
        self.Y = []
