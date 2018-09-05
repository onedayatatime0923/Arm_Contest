
from visualizer.base_visualizer import BaseVisualizer


class SensorVisualizer(BaseVisualizer):
    def __init__(self, repr = ['A X', 'A Y', 'A Z', 'G X', 'G Y', 'G Z', 'M X', 'M Y', 'M Z']):
        self.repr = repr
    def __call__(self, data):
        message = '\x1b[2K\r'
        for n in range(len(self.repr)):
            message += '{}: {:>8.4f} | '.format(
                self.repr[n], float(data[n]))
        print(message)

if __name__ == '__main__':
    v = SensorVisualizer()


