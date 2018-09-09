
import numbers
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 

class Painter():
    def __init__(self, repr = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Mx', 'My', 'Mz', 'Q1', 'Q2', 'Q3', 'Q4', 'Y', 'P', 'R'], display= None, memorySize = 10, ylim = [2, -2]):
        self.n = len(repr)
        self.repr = repr
        if display is None:
            self.display = list(range(self.n))
        else:
            self.display = display
        self.memorySize = memorySize
        if isinstance(ylim, numbers.Number):
            self.ylim = (-ylim, ylim)
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
        for i in self.display:
            self.line[i] = plt.plot(self.data[i][-self.memorySize:], label = self.repr[i])[0]
        plt.xlim((0, self.memorySize))
        plt.ylim(self.ylim)
        plt.legend(loc='upper right')

    def _update(self, index): 
        print(self.data.shape)
        print(self.data[:,-self.memorySize:])
        input()
        for i in self.display:
            self.line[i].set_ydata(self.data[i][-self.memorySize:])

    
if __name__ == '__main__':
    p = Painter()
