
import numpy as np
assert np


class Visualizer():
    def __init__(self, name = ['A X', 'A Y', 'A Z', 'G X', 'G Y', 'G Z', 'M X', 'M Y', 'M Z']):
        self.name = name
    def __call__(self, data):
        message = '\x1b[2K\r'
        for n in range(len(self.name)):
            message += '{}: {:>8.4f} | '.format(
                self.name[n], float(data[n]))
        print(message, flush = True)

if __name__ == '__main__':
    v = Visualizer()


