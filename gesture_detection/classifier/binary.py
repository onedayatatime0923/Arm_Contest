
import numpy as np
import numpy.linalg as LA

class Classifier:
    def __init__(self, threshold = 40, index = [3,4,5], nStep = 8):
        self.threshold = threshold
        self.index = np.array(index)
        self.nStep = nStep
        self.data = [np.zeros(len(index)) for i in range(nStep)]

    def __call__(self, data):
        data = data[self.index]
        self.data = self.data[1:] + [data]
        if all([ LA.norm(i) < self.threshold for i in self.data]):
            return False
        else:
            return True

if __name__ == '__main__':
    c = Classifier()
