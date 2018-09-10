


from options import RecorderOptions
from visualizer import SensorVisualizer
from sensor import Sensor
from filter import Filter
from recorder import Recorder
from classifier import ClassifierBinary

parser = RecorderOptions()
opt = parser.parse()

raw_input("Enter to start")
sensor = Sensor(opt.n, opt.port, opt.freq)
filter = Filter(opt.n, opt.n)
visualizer = SensorVisualizer(repr = opt.repr)
recorder = Recorder(opt)
classifier = ClassifierBinary(opt.threshold, opt.index, opt.nStep)

print("action: {}".format(opt.action))


def main(i):
    lastSignal = False
    moveCount = 0
    stopCount = 0
    while True:
        data = sensor.read()
        data = filter.update(data)
        signal = classifier(data)
        if signal:
            moveCount += 1
            print('move', moveCount)
            recorder.label(data)
        else:
            stopCount += 1
            print('stop', stopCount)
            if lastSignal == True:
                pass
                recorder.dump_action()
        lastSignal = signal
        #visualizer(data)
    
if(__name__ == '__main__'):
    main()
