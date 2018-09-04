
import sys
sys.path.append('./')
from options.base_options import BaseOptions


class RecorderOptions(BaseOptions):
    def __init__(self):
        BaseOptions.__init__(self)
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # ---------- Define Recorder ---------- #
        parser.add_argument('--mode', type=str, choices = ['stop', 'move'],
                default = 'stop')
        parser.add_argument('--split', type=str, choices = ['train', 'val'],
                default = 'train')
        parser.add_argument('--recordInterval', type=int,
                default = 100)
        return parser
