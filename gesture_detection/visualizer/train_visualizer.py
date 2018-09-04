
from visualizer.base_visualizer import BaseVisualizer

class TrainVisualizer(BaseVisualizer):
    def __init__(self, opt, dataset):
        super(TrainVisualizer, self).__init__(opt.batchSize, len(dataset), opt.logPath)
        # init argument:  stepSize, totalSize, displayWidth, logPath):

