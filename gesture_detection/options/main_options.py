
import sys
sys.path.append('./')
from options.base_options import BaseOptions


class MainOptions(BaseOptions):
    def __init__(self):
        BaseOptions.__init__(self)
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # ---------- Define Device ---------- #
        parser.add_argument('--n', type=int, default = 16)
        parser.add_argument('--port', type=str, default = '/dev/cu.usbmodem1413')
        parser.add_argument('--freq', type=int, default = 115200)
        parser.add_argument('--repr', type=str, nargs = 16, 
                default = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Mx', 'My', 'Mz', 'Q1', 'Q2', 'Q3', 'Q4', 'Y', 'P', 'R'])
        # ---------- Define Painter ---------- #
        parser.add_argument('--display', type=int, nargs = '+', 
                default = list(range(16)))
        parser.add_argument('--memorySize', type=int, default = 10)
        parser.add_argument('--ylim', type=int, default = 200)
        # ---------- Experiment Setting ---------- #
        parser.set_defaults(name= 'main')
        return parser
