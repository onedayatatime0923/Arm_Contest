
import os
from dtw import dtw
import numpy as np
from numpy import linalg as LA

assert dtw

class Classifier:
    def __init__(self, threshold = 98, noOps = 'None'):
        self.threshold = threshold
        self.noOps = noOps
        self.action = None
        self.data = None
    def read(self):
        action = []
        data = []
        with open("./utils/vocabulary/record.txt", 'r') as f:
            for line in f:
                act = line.strip()
                path = os.path.join('data',act)
                fileList = os.listdir(path) if os.path.exists(path) else []
                for f in fileList:
                    action.append(act)
                    data.append(np.array([i for i in np.load(os.path.join(path,f))[0]]))
        self.action = action
        self.data = data
    def predict(self, target):
        target = np.array(target)
        score = []
        for d in self.data:
            score.append(dtw(d, target, dist=lambda x, y: LA.norm(x - y, ord=1))[0])
        score = np.array(score)
        print(min(np.array(score)))
        if min(np.array(score)) < self.threshold:
            return self.action[np.argmin(score)]
        else:
            return self.noOps
