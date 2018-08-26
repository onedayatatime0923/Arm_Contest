
import numpy as np
assert np


class Visualizer():
    def __init__(self):
        pass
    def __call__(self, data):
        message = '\x1b[2K\r'
        print(data)
        input()
        for n in data:
            print(n, data[n])
            message += '{} X: {} Y: {} Z:{} | '.format(
                n, data[n][0], data[n][1], data[n][2])

        print(message, flush = True)

if __name__ == '__main__':
    v = Visualizer()


