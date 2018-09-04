
from visualizer.base_visualizer import BaseVisualizer

class TrainVisualizer(BaseVisualizer):
    def __init__(self, opt, dataset):
        BaseVisualizer.__init__(self, opt.batchSize, len(dataset), opt.logPath)
        # init argument:  stepSize, totalSize, displayWidth, logPath):

