
import multiprocessing as mp
import torch
import numpy as np
from torch.autograd import Variable

from options import MainOptions
from visualizer import SensorVisualizer, Painter
from sensor import Sensor
from filter import Filter
from models import createModel
assert mp and np and torch and Variable

parser = MainOptions()
opt = parser.parse()

sensor = Sensor(opt.port)
filter = Filter(opt.n, opt.n)
visualizer = SensorVisualizer(name = opt.repr)
painter = Painter(name = opt.repr, verbose = opt.display, memorySize = opt.memorySize, ylim = opt.ylim )

# set model

model = createModel(opt)
model.setup(opt)
model.eval()

def main():
    counter = 0
    th = 3.4
    gesture = []
    while True:
        data = filter.update(sensor.read())
        #print(data)
        #x = Variable(torch.FloatTensor(data)).squeeze(1).unsqueeze(0)
        #print(model.forward(x))
        threshold = data[0] 
        print(data[0])
        if(threshold >= th or threshold <= -th):
            gesture.append(data)
            pass
            #print("Move")
        else:
            counter += 1
            if(len(gesture) != 0):
                print("Dump file {}...".format(counter))
                #np.save("data/gesture/{}.npy".format(counter), np.array(gesture))
            gesture = []
            #print("Stop", counter)

        #visualizer(data)
        #painter(data)

main()
#p1 = mp.Process(target=main)
#p1.start()
#painter.plot()
