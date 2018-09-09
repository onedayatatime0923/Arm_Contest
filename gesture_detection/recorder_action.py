
import torch
from torch.autograd import Variable


from options import TrainOptions
from visualizer import SensorVisualizer
from sensor import Sensor
from filter import Filter
from recorder import Recorder
from utils import convert
from models import createModel

parser = TrainOptions()
opt = parser.parse()

sensor = Sensor(opt.n, opt.port, opt.freq)
filter = Filter(opt.n, opt.n)
visualizer = SensorVisualizer(repr = opt.repr)
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
        x = Variable(convert(torch.FloatTensor(data), opt.n))
        signal = model.predict(x)
        if signal:
            pass
            #recorder.label(data)
        else:
            count += 1
            print('stop', count)
            if lastSignal == True:
                recorder.dump_action()
        lastSignal = signal
        #visualizer(data)
    
if(__name__ == '__main__'):
    main()
