
import torch
from visualizer.base_visualizer import BaseVisualizer

class TrainVisualizer(BaseVisualizer):
    def __init__(self, opt, dataset):
        super(TrainVisualizer, self).__init__(opt.batchSize, len(dataset), opt.logPath)
        # init argument:  stepSize, totalSize, displayWidth, logPath):
    def construct_device(self):
        # set gpu ids
        if self.opt.gpuIds[0] != -1:
            self.opt.device = torch.device(self.opt.gpuIds[0])
        else:
            self.opt.device = torch.device('cpu')

