
import torch
from torch.autograd import Variable


from options import MainOptions
from visualizer import SensorVisualizer, Painter
from sensor import Sensor
from filter import Filter
from recorder import Recorder
from utils import convert
from models import createModel

parser = MainOptions()
opt = parser.parse()

sensor = Sensor(opt.port)
filter = Filter(opt.n, opt.n)
visualizer = SensorVisualizer(repr = opt.repr)
painter = Painter(repr = opt.repr, display = opt.display, memorySize = opt.memorySize, ylim = opt.ylim )
recorder = Recorder(opt)

print("action: {}".format(opt.action))

model = createModel(opt)
model.setup(opt)
model.eval()

def main():
    lastSignal = False
    count = 0
    while True:
        data = sensor.read()
        data = filter.update(data)
        print(data)
        x = Variable(convert(torch.FloatTensor(data)))
        signal = model.predict(x)
        if signal:
            recorder.label(data)
        else:
            count += 1
            print('stop', count)
            if lastSignal == True:
                recorder.dump_action()
        lastSignal = signal
        #visualizer(data)
    
if(__name__ == '__main__'):
    main()
