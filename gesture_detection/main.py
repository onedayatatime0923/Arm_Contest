
import multiprocessing as mp
import threading as td
assert mp and td

from options import MainOptions
from visualizer import SensorVisualizer, Painter
from sensor import Sensor
from filter import Filter

parser = MainOptions()
opt = parser.parse()

sensor = Sensor(opt.n, opt.port, opt.freq)
filter = Filter(opt.n, opt.n)
visualizer = SensorVisualizer(repr = opt.repr)
painter = Painter(repr = opt.repr, display = opt.display, memorySize = opt.memorySize, ylim = opt.ylim)

def main():
    while True:
        data = sensor.read()
        data = filter.update(data)
        visualizer(data)
        painter(data)
    
if(__name__ == '__main__'):
    painter.plot()
    main()
    '''
    p1 = td.Thread(target=main)
    p1.start()
    '''
