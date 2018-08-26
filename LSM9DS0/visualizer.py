
import numpy as np
assert np


class Visualizer():
    def __init__(self):
        pass
    def __call__(self, data):
        message = '\x1b[2K\r'
        for n in data:
            message += '{} X: {:>8.4f} Y: {:>8.4f} Z:{:>8.4f} | '.format(
                n, data[n][0], data[n][1], data[n][2])
        print(message, flush = True)

if __name__ == '__main__':
    v = Visualizer()


