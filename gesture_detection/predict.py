


from options import MainOptions
from sensor import Sensor
from filter import Filter
from visualizer import SensorVisualizer
from speech import Speech
from classifier_dtw import Classifier

parser = MainOptions()
opt = parser.parse()
sensor = Sensor(opt.n, opt.port, opt.freq)
filter = Filter(opt.n, opt.n)
visualizer = SensorVisualizer(repr = opt.repr)
speech = Speech()
classifier = Classifier(opt.threshold)



def main():
    target = []
    while True:
        data = sensor.read()
        data = filter.update(data)
        target.append(data)
        operate = classifier.predict(target)
        print(operate)
        if operate != 'None':
            speech(operate)
            target = []
        if len(target) > opt.maxLen:
            print(len(target))
            target = target[-opt.maxLen:]
        #visualizer(data)
    
if(__name__ == '__main__'):
    raw_input("Enter to start")
    main()
