
import sys
sys.path.append('./')
from options.base_options import BaseOptions


class MainOptions(BaseOptions):
    def __init__(self):
        BaseOptions.__init__(self)
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # ---------- Define Mode ---------- #
        parser.set_defaults(mode = 'train')
        # ---------- Define Network ---------- #
        parser.set_defaults(pretrained = True)
        return parser
    def parse(self):
        BaseOptions.model_parse(self)

        return self.opt
