
from visualizer.base_visualizer import BaseVisualizer

class ModelVisualizer(BaseVisualizer):
    def __init__(self, opt, dataset):
        BaseVisualizer.__init__(self, opt.batchSize, len(dataset), opt.displayWidth, opt.logPath)
        # init argument:  stepSize, totalSize, displayWidth, logPath):

