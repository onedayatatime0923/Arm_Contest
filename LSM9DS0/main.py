
import threading as td
import numpy as np

from visualizer import Visualizer
from painter import Painter
from sensor import Sensor
#from filter import Filter

sensor = Sensor("/dev/cu.usbmodem1413")
#filter = Filter(6,6)
visualizer = Visualizer()
painter = Painter()

def main():
    while True:
        data = sensor.read()
        #data = filter.update(sensor.read())
        visualizer(data)
        data['G'] = np.array([0,0,0])
        painter(data)

td.Thread(target=main).start()
painter.plot()
