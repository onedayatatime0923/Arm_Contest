
import torch
import sys
sys.path.append('./')
from options.base_options import BaseOptions


class MainOptions(BaseOptions):
    def __init__(self):
        BaseOptions.__init__(self)
        #super(MainOptions, self).__init__()
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # ---------- Define Network ---------- #
        parser.set_defaults(pretrained = True)
        return parser
    def parse(self):
        # gather options
        self.gather_options()
        if self.opt.mode == 'train' and not self.opt.resume:
            self.construct_checkpoints(creatDir = True)
        elif self.opt.mode == 'train' and self.opt.resume:
            self.construct_checkpoints(creatDir = False)
        elif self.opt.mode == 'test':
            self.construct_checkpoints(creatDir = False)
            self.construct_outputPath()

        # continue to train
        if self.opt.mode == 'train' and self.opt.resume:
            self.load_options('opt.txt')

        # print options
        self.construct_message()
        if self.opt.mode == 'train' and not self.opt.resume:
            self.save_options('opt.txt')
        if self.opt.mode == 'test':
            self.save_options('test_opt.txt')
        self.print_options()

        # set gpu ids
        self.construct_device()

        return self.opt
    def construct_device(self):
        # set gpu ids
        if self.opt.gpuIds[0] != -1:
            self.opt.device = torch.device(self.opt.gpuIds[0])
        else:
            self.opt.device = torch.device('cpu')
