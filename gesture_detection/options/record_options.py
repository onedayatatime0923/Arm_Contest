
import sys
sys.path.append('./')
from options.base_options import BaseOptions


class RecorderOptions(BaseOptions):
    def __init__(self):
        BaseOptions.__init__(self)
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        return parser
