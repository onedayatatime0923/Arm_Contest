
import threading as td

from visualizer import Visualizer
from painter import Painter
from sensor import Sensor
from filter import Filter

name = ['Accel X', 'Accel Y', 'Accel Z', 'Gyro X', 'Gyro Y', 'Gyro Z', 'Magnetic X', 'Magnetic Y', 'Magnetic Z']
sensor = Sensor("COM3")
filter = Filter(9,9)
visualizer = Visualizer( name = name)
painter = Painter(name = name)

def main():
    while True:
        data = filter.update(sensor.read())
        visualizer(data)
        painter(data)

td.Thread(target=main).start()
painter.plot()
