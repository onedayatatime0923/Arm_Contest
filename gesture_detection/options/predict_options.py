
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
        parser.add_argument('--threshold', type=float, default = 100.)
        parser.add_argument('--maxLen', type=int, default = 50)
        return parser
