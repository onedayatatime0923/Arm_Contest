
from visualizer import Visualizer
from painter import Painter
from sensor import Sensor
from filter import Filter

sensor = Sensor()
filter = Filter(6,6)
visualizer = Visualizer()
painter = Painter()
painter.plot()

while True:
    data = filter.update(sensor.read())
    visualizer(painter)
    painter(data)
