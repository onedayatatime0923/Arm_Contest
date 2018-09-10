
import numpy as np
import numpy.linalg as LA

from options import ActionOptions
from visualizer import SensorVisualizer
from sensor import Sensor
from filter import Filter
from recorder import Recorder

parser = ActionOptions()
opt = parser.parse()

raw_input("Enter to start")
sensor = Sensor(opt.n, opt.port, opt.freq)
filter = Filter(opt.n, opt.n)
visualizer = SensorVisualizer(repr = opt.repr)
recorder = Recorder(opt)

#print("action: {}".format(opt.action))


def main():
    target = [np.zeros(len(opt.index)) for i in range(opt.nStep)]
    moveCount = 0
    stopCount = 0
    while True:
        data = sensor.read()
        data = filter.update(data)
        visualizer(data)
        data = data[np.array(opt.index)]
        print(LA.norm(data))
        target = target[1:] + [data]
        if all([ LA.norm(i) < opt.threshold for i in target]):
            stopCount += 1
            print('stop', stopCount)
        else:
            moveCount += 1
            print('move', moveCount)
    
if(__name__ == '__main__'):
    main()
