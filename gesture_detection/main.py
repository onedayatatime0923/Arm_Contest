
import multiprocessing as mp
import torch
from torch.autograd import Variable

from options import MainOptions
from visualizer import SensorVisualizer, Painter
from sensor import Sensor
from filter import Filter
from models import createModel
assert mp

parser = MainOptions()
opt = parser.parse()

sensor = Sensor(opt.port)
filter = Filter(opt.n, opt.n)
visualizer = SensorVisualizer(name = opt.repr)
painter = Painter(name = opt.repr, verbose = opt.display, memorySize = opt.memorySize, ylim = opt.ylim )

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

main()
#p1 = mp.Process(target=main)
#p1.start()
#painter.plot()
