
import argparse, os
from ast import literal_eval
import torch


class BaseOptions():
    def __init__(self):
        self.parser = None
        self.opt = None
        self.message = None

    def initialize(self, parser):
        # ---------- Define Device ---------- #
        parser.add_argument('--port', type=str, default = '/dev/cu.usbmodem1413')
        parser.add_argument('--n', type=int, default = 9)
        parser.add_argument('--repr', type=str, nargs = 9, 
                default = ['Accel X', 'Accel Y', 'Accel Z', 'Gyro X', 'Gyro Y', 
                    'Gyro Z', 'Magnetic X', 'Magnetic Y', 'Magnetic Z'])
        # ---------- Define Painter ---------- #
        parser.add_argument('--display', type=int, nargs = '+', 
                default = list(range(9)))
        parser.add_argument('--memorySize', type=int, default = 10)
        parser.add_argument('--ylim', type=int, default = 2)
        # ---------- Experiment Setting ---------- #
        parser.add_argument('--name', type=str,default = 'Sensor',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', 
                            help='models are saved here')
        return parser

    def gather_options(self):
        # initialize parser with basic options
        parser = argparse.ArgumentParser(
                description='PyTorch Segmentation Adaptation')

        parser = self.initialize(parser)
        self.parser = parser
        self.opt = parser.parse_args()

    def construct_checkpoint(self,creatDir = True):
        if creatDir:
            index = 0
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            while os.path.exists(path):
                path = os.path.join(self.opt.checkpoints_dir, '{}_{}'.format(self.opt.name,index))
                index += 1
            self.opt.expPath = path
            self.opt.logPath = os.path.join(self.opt.expPath, 'log')
            self.opt.modelPath = os.path.join(self.opt.expPath, 'model')
            os.makedirs(self.opt.expPath)
            os.makedirs(self.opt.logPath)
            os.makedirs(self.opt.modelPath)
        else:
            self.opt.expPath = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            self.opt.logPath = os.path.join(self.opt.expPath, 'log')
            self.opt.modelPath = os.path.join(self.opt.expPath, 'model')
            assert( os.path.exists(self.opt.expPath) and os.path.exists(self.opt.logPath) and os.path.exists(self.opt.modelPath) )


    def load_options(self, path):
        # load from the disk
        file_name = os.path.join(self.opt.expPath, path)
        with open(file_name, 'rt') as opt_file:
            for line in opt_file:
                if line == '-------------------- Options ------------------\n' or \
                   line == '-------------------- End ----------------------\n':
                       continue
                line = line.split('[default: ',1)[0].strip()
                arg, val = line.split(': ',1)
                # only resume has None type so yet it would't be saved
                if hasattr(self.opt, arg):
                    valType = type(getattr(self.opt, arg))
                    if (valType == list) or (valType == tuple) or (valType == bool):
                        setattr(self.opt, arg, literal_eval(val))
                    else:
                        setattr(self.opt, arg, (type(getattr(self.opt, arg)))(val))

    def construct_message(self):
        message = ''
        message += '-------------------- Options ------------------\n'
        for k, v in sorted(vars(self.opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '-------------------- End ----------------------'
        self.message = message

    def save_options(self, path):
        # save to the disk
        file_name = os.path.join(self.opt.expPath, path)
        with open(file_name, 'wt') as opt_file:
            opt_file.write(self.message)
            opt_file.write('\n')
        
    def print_options(self):
        print(self.message)

    def construct_device(self):
        # set gpu ids
        if self.opt.gpuIds[0] != -1:
            self.opt.device = torch.device(self.opt.gpuIds[0])
        else:
            self.opt.device = torch.device('cpu')

    def parse(self):
        # gather options
        self.gather_options()
        self.construct_checkpoint(creatDir = True)

        # print options
        self.construct_message()
        self.save_options('opt.txt')
        self.print_options()

        return self.opt

    def update(self):
        self.construct_message()
        self.save_options()
