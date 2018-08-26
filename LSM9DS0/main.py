
from visualizer import Visualizer
from painter import Painter
from sensor import Sensor
import multiprocessing
#from filter import Filter

sensor = Sensor("/dev/cu.usbmodem1413")
#filter = Filter(6,6)
visualizer = Visualizer()
painter = Painter()
process = multiprocessing.Process(target=painter.plot,args=())
process.start()

def main():
    while True:
        data = sensor.read()
        #data = filter.update(sensor.read())
        visualizer(data)
        #painter(data)
main()
