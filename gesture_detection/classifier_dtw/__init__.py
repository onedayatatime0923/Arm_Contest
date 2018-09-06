
import os
from dtw import dtw
import numpy as np

assert dtw

class Classifier:
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
                    data.append(np.load(os.path.join(path,f)))
        self.action = action
        self.data = data

