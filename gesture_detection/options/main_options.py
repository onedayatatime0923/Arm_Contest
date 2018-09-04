
import sys
sys.path.append('./')
from options.base_options import BaseOptions


class MainOptions(BaseOptions):
    def __init__(self):
        BaseOptions.__init__(self)
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # ---------- Define Mode ---------- #
        parser.set_defaults(mode = 'use')
        # ---------- Define Network ---------- #
        parser.set_defaults(pretrained = True)
        return parser
    def parse(self):
        # gather options
        self.gather_options()
        self.construct_checkpoints(creatDir = True)

        # print options
        self.construct_message()
        self.save_options('opt.txt')
        self.print_options()

        # set gpu ids
        self.construct_device()

        return self.opt
