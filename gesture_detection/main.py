
import multiprocessing as mp

from options import SensorOptions
from visualizer import SensorVisualizer, Painter
from sensor import Sensor
from filter import Filter


parser = SensorOptions()
opt = parser.parse()

sensor = Sensor(opt.port)
filter = Filter(opt.n, opt.n)
visualizer = SensorVisualizer(name = opt.name)
painter = Painter(name = opt.name, verbose = opt.verbose, memorySize = opt.memorySize, ylim = opt.ylim )

def main():
    while True:
        data = filter.update(sensor.read())
        visualizer(data)
        painter(data)

p1 = mp.Process(target=main)
p1.start()
painter.plot()
