import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 

class Painter():
    def __init__(self, name = ['Accel X', 'Accel Y', 'Accel Z', 'Gyro X', 'Gyro Y', 'Gyro Z', 'Magnetic X', 'Magnetic Y', 'Magnetic X'], memorySize= 10, frames = 1000):
        self.n = len(name)
        self.memorySize = memorySize
        self.data = np.zeros((self.n, self.memorySize))
        self.frames = frames

        self.fig = None
        self.animation = None
        self.line = [None for i in range(self.n)]

    def __call__(self, data):
        data = np.concatenate(list(data.values()), 0)
        self.data = np.append(self.data, data)

    def plot(self):
        fig = plt.figure() 
        self.animation = animation.FuncAnimation(
                fig=fig,
                func=self._update,
                frames=self.frames,
                init_func=self._init,
                interval=20,
                blit=False)
        plt.show()

    def save(self, path):
        self.ani.save(path, fps=30, extra_args=['-vcodec', 'libx264'])

    def _init(self):
        for i in range(self.n):
            self.line[i] = plt.plot(self.data[i][:-self.memorySize], label = self.name[i])[0]

    def _update(self, index): 
        for i in range(self.n):
            self.line[i].set_ydata(self.data[i][:-self.memorySize])

if __name__ == '__main__':
    p = Painter()
