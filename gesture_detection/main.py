
import multiprocessing as mp
import torch
from torch.nn import Variable

from options import SensorOptions
from visualizer import SensorVisualizer, Painter
from sensor import Sensor
from filter import Filter
from models import createModel
assert mp

parser = SensorOptions()
opt = parser.parse()

sensor = Sensor(opt.port)
filter = Filter(opt.n, opt.n)
visualizer = SensorVisualizer(name = opt.name)
painter = Painter(name = opt.name, verbose = opt.verbose, memorySize = opt.memorySize, ylim = opt.ylim )

# set model

model = createModel(opt)
model.setup(opt)

def main():
    while True:
        data = filter.update(sensor.read())
        x = Variable(torch.FloatTensor(data))
        print(model.forward(x))

        visualizer(data)
        painter(data)

#p1 = mp.Process(target=main)
#p1.start()
painter.plot()
