
import sys
sys.path.append('./')
from options.base_options import BaseOptions


class SensorOptions(BaseOptions):
    def __init__(self):
        super(SensorOptions, self).__init__()
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        return parser
