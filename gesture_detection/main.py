
import threading as td

from options import TrainOptions
from visualizer.visualizer import Visualizer
from painter import Painter
from sensor import Sensor
from filter import Filter


parser = TrainOptions()
opt = parser.parse()

sensor = Sensor(opt.port)
filter = Filter(opt.n, opt.n)
visualizer = Visualizer( name = opt.name)
painter = Painter(name = opt.name, verbose = opt.verbose, memorySize = opt.memorySize, ylim = opt.ylim )

def main():
    while True:
        data = filter.update(sensor.read())
        visualizer(data)
        painter(data)

td.Thread(target=main).start()
painter.plot()
