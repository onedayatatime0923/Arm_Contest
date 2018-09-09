
from visualizer.base_visualizer import BaseVisualizer


class SensorVisualizer(BaseVisualizer):
    def __init__(self, repr = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Mx', 'My', 'Mz', 'Q1', 'Q2', 'Q3', 'Q4', 'Y', 'P', 'R']):
        self.repr = repr
    def __call__(self, data):
        assert len(self.repr) == len(data)
        message = '\x1b[2K\r'
        for n in range(len(self.repr)):
            message += '{}: {:>8.4f} | '.format(
                self.repr[n], float(data[n]))
        print(message)

if __name__ == '__main__':
    v = SensorVisualizer()


