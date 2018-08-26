
import numpy as np
assert np


class Visualizer():
    def __init__(self, name = ['Accel X', 'Accel Y', 'Accel Z', 'Gyro X', 'Gyro Y', 'Gyro Z'], memorySize = 10):
        self.name = name
        self.n = len(name)
        self.memorySize = memorySize
        self.data = np.empty((self.n))
    def __call__(self, data):
        message = '\x1b[2K\r'
        for i in range(self.n):
            print(self.name[i], data[i])
            message += '{}: {:.4f} | '.format(
                self.name[i], float(data[i]))
        self.data = np.append(self.data, data, axis = 1)

        print(message, flush = True)

if __name__ == '__main__':
    v = Visualizer()


