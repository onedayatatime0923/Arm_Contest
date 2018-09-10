
import sys
sys.path.append('./')
from options.base_options import BaseOptions


class ActionOptions(BaseOptions):
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
        # ---------- Define Recorder ---------- #
        '''
        parser.add_argument('--action', type=str,
                default = 'stop')
        '''
        parser.add_argument('--dataDir', type=str, default='./data', 
                            help='models are saved here')
        # ---------- Define Painter ---------- #
        parser.add_argument('--display', type=int, nargs = '+', 
                default = list(range(16)))
        parser.add_argument('--memorySize', type=int, default = 10)
        parser.add_argument('--ylim', type=int, default = 200)
        # ---------- Define Parameters ---------- #
        parser.add_argument('--threshold', type=float, default = 40)
        parser.add_argument('--index', type=int, nargs = '*',
                default = [3,4,5])
        parser.add_argument('--nStep', type=int, default = 8)
        # ---------- Experiment Setting ---------- #
        parser.set_defaults(name= 'action')
        return parser

    def parse(self):
        # gather options
        self.gather_options()
        self.construct_checkpoints(creatDir = True)
        #self.construct_splitDir()
        #self.construct_actionDir()

        # print options
        self.construct_message()
        self.save_options('opt.txt')
        self.print_options()

        return self.opt
