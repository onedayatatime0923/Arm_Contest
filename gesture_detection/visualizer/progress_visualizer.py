
from visualizer.base_visualizer import BaseVisualizer

class ProgressVisualizer(BaseVisualizer):
    def __init__(self, opt, dataset):
        BaseVisualizer.__init__(self, opt.batchSize, len(dataset))
        # init argument:  stepSize, totalSize, displayWidth, logPath):

