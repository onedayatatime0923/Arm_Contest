

from options import SensorOptions
from visualizer import SensorVisualizer, Painter
from sensor import Sensor
from filter import Filter
from recoder import Recorder

parser = SensorOptions()
opt = parser.parse()

sensor = Sensor(opt.port)
filter = Filter(opt.n, opt.n)
visualizer = SensorVisualizer(name = opt.name)
painter = Painter(name = opt.name, verbose = opt.verbose, memorySize = opt.memorySize, ylim = opt.ylim )
recorder = Recorder()

counter = 0

def main():
    while True:
        data = sensor.read()
        recorder(data)
        visualizer(data)
    
if(__name__ == '__main__'):
    main()
