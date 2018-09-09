
import sys
sys.path.append('./')
from options.base_options import BaseOptions


class MainOptions(BaseOptions):
    def __init__(self):
        BaseOptions.__init__(self)
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # ---------- Experiment Setting ---------- #
        parser.set_defaults(name= 'main')
        return parser
    def parse(self):
        BaseOptions.parse(self)

        return self.opt
