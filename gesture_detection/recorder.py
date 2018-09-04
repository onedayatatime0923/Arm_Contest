import numpy as np
from visualizer import Visualizer
from sensor import Sensor
#from filter import Filter

name = ['Accel X', 'Accel Y', 'Accel Z', 'Gyro X', 'Gyro Y', 'Gyro Z', 'Magnetic X', 'Magnetic Y', 'Magnetic Z']
sensor = Sensor('/dev/cu.usbmodem1413')
visualizer = Visualizer(name = name)
record_cycle = 100

class Recorder():
    def __init__(self):
        self.X = []
        self.Y = []

    def labeling(self, data, mode='stop'):
        if mode == 'stop':
            self.X.append(data)
            self.Y.append(0)
        elif mode == 'move':
            self.X.append(data)
            self.Y.append(1)
    def dump_data(self):
        np.save("./data.npy", np.array([self.X, self.Y]))
        self.X = []
        self.Y = []


def main():
    rec = Recorder()
    counter = 0
    mode = 'stop'
    while True:
        if counter % record_cycle == 0:
            mode = input("Enter mode: ")
        if(mode == 'dump'):
            rec.dump_data()
        else:
            counter += 1
            data = sensor.read()
            visualizer(data)
            rec.labeling(data, mode)
    
if(__name__ == '__main__'):
    main()

    
