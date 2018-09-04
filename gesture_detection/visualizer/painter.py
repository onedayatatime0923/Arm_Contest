
import numbers
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 

class Painter():
    def __init__(self, name = ['Accel X', 'Accel Y', 'Accel Z', 'Gyro X', 'Gyro Y', 'Gyro Z', 'Magnetic X', 'Magnetic Y', 'Magnetic Z'], verbose = None, memorySize = 10, ylim = [2, -2]):
        self.n = len(name)
        self.name = name
        if verbose is None:
            self.verbose = list(range(self.n))
        else:
            self.verbose = verbose
        self.memorySize = memorySize
        if isinstance(ylim, numbers.Number):
            self.ylim = (ylim, ylim)
        else:
            self.ylim = ylim
        self.data = np.zeros((self.n, self.memorySize))

        self.animation = None
        self.line = [None for i in range(self.n)]

    def __call__(self, data):
        self.data = np.append(self.data, data, 1)

    def plot(self):
        fig = plt.figure() 
        self.animation = animation.FuncAnimation(
                fig=fig,
                func=self._update,
                init_func=self._init,
                interval=20,
                blit=False)
        plt.show()

    def save(self, path):
        self.animation.save(path, fps=30, extra_args=['-vcodec', 'libx264'])

    def _init(self):
        for i in self.verbose:
            self.line[i] = plt.plot(self.data[i][-self.memorySize:], label = self.name[i])[0]
        plt.xlim((0, self.memorySize))
        plt.ylim(self.ylim)
        plt.legend(loc='upper right')

    def _update(self, index): 
        for i in self.verbose:
            self.line[i].set_ydata(self.data[i][-self.memorySize:])

    
if __name__ == '__main__':
    p = Painter()