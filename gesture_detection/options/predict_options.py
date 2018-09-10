
import sys
sys.path.append('./')
from options.base_options import BaseOptions


class PredictOptions(BaseOptions):
    def __init__(self):
        BaseOptions.__init__(self)
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # ---------- Define Device ---------- #
        parser.add_argument('--n', type=int, default = 16)
        parser.add_argument('--port', type=str, default = '/dev/cu.usbmodem1413')
        parser.add_argument('--freq', type=int, default = 57600)
        parser.add_argument('--repr', type=str, nargs = 16, 
                default = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Mx', 'My', 'Mz', 'Q1', 'Q2', 'Q3', 'Q4', 'Y', 'P', 'R'])
        # ---------- Define Parameters ---------- #
        parser.add_argument('--binaryThreshold', type=float, default = 40)
        parser.add_argument('--actionThreshold', type=float, default = 98)
        parser.add_argument('--index', type=int, nargs = '*',
                default = [3,4,5])
        parser.add_argument('--nStep', type=int, default = 8)
        # ---------- Experiment Setting ---------- #
        parser.set_defaults(name= 'predict')
        return parser
