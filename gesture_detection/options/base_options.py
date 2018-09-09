
import torch
import argparse, os
from ast import literal_eval


class BaseOptions():
    def __init__(self):
        self.parser = None
        self.opt = None
        self.message = None

    def initialize(self, parser):
        # ---------- Define Mode ---------- #
        parser.add_argument('--mode', type=str, choices = ['train','test'],
                default = 'train', help="Model Mode")
        # ---------- Define Device ---------- #
        parser.add_argument('--n', type=int, default = 16)
        parser.add_argument('--port', type=str, default = '/dev/cu.usbmodem1413')
        parser.add_argument('--freq', type=int, default = 921600)
        parser.add_argument('--repr', type=str, nargs = 16, 
                default = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Mx', 'My', 'Mz', 'Q1', 'Q2', 'Q3', 'Q4', 'Y', 'P', 'R'])
        # ---------- Define Recorder ---------- #
        parser.add_argument('--action', type=str,
                default = 'stop')
        parser.add_argument('--dataDir', type=str, default='./data', 
                            help='models are saved here')
        parser.add_argument('--split', type=str, choices = ['train', 'val'],
                default = 'train')
        parser.add_argument('--recordInterval', type=int,
                default = 100)
        # ---------- Define Network ---------- #
        parser.add_argument('--gpuIds', type=int, nargs = '+', default=[-1], help='gpu ids: e.g. 0, 0 1, 0 1 2,  use -1 for CPU')
        parser.add_argument('--model', type=str, default = 'binary',
                            help="Method Name")
        parser.add_argument('--ncf', type=int, default= 64, help= 'number of filters')
        parser.add_argument('--pretrained', action = 'store_true', help='whether to use pretrained model')
        parser.add_argument('--pretrainedRoot', type = str, default = 'pretrained/', help='path to load pretrained model')
        # ---------- Define Painter ---------- #
        parser.add_argument('--display', type=int, nargs = '+', 
                default = list(range(16)))
        parser.add_argument('--memorySize', type=int, default = 10)
        parser.add_argument('--ylim', type=int, default = 200)
        # ---------- Whether to Resume ---------- #
        parser.add_argument("--resume", action = 'store_true',
                            help="whether to resume")
        parser.add_argument("--resumeName", type=str, default='latest',
                            help="model(pth) path, set to latest to use latest cached model")
        # ---------- Experiment Setting ---------- #
        parser.add_argument('--name', type=str,default = 'Sensor',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--checkpointsDir', type=str, default='./checkpoints', 
                            help='models are saved here')
        parser.add_argument('--verbose', action='store_true', 
                            help='if specified, print more debugging information')
        return parser

    def gather_options(self):
        # initialize parser with basic options
        parser = argparse.ArgumentParser(
                description='PyTorch Segmentation Adaptation')

        parser = self.initialize(parser)
        self.parser = parser
        self.opt = parser.parse_args()

    def construct_checkpoints(self,creatDir = True):
        if creatDir:
            index = 0
            path = os.path.join(self.opt.checkpointsDir, self.opt.name)
            while os.path.exists(path):
                path = os.path.join(self.opt.checkpointsDir, '{}_{}'.format(self.opt.name,index))
                index += 1
            self.opt.expPath = path
            self.opt.logPath = os.path.join(self.opt.expPath, 'log')
            self.opt.modelPath = os.path.join(self.opt.expPath, 'model')
            os.makedirs(self.opt.expPath)
            os.makedirs(self.opt.logPath)
            os.makedirs(self.opt.modelPath)
        else:
            self.opt.expPath = os.path.join(self.opt.checkpointsDir, self.opt.name)
            self.opt.logPath = os.path.join(self.opt.expPath, 'log')
            self.opt.modelPath = os.path.join(self.opt.expPath, 'model')
            assert( os.path.exists(self.opt.expPath) and os.path.exists(self.opt.logPath) and os.path.exists(self.opt.modelPath) )

    def construct_splitDir(self):
        self.opt.splitDir = os.path.join(self.opt.dataDir, self.opt.split)
        if not os.path.exists(self.opt.splitDir):
            os.makedirs(self.opt.splitDir)

    def construct_actionDir(self):
        self.opt.actionDir = os.path.join(self.opt.dataDir, self.opt.action)
        if not os.path.exists(self.opt.actionDir):
            os.makedirs(self.opt.actionDir)

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
        self.construct_checkpoints(creatDir = True)

        # print options
        self.construct_message()
        self.save_options('opt.txt')
        self.print_options()

        return self.opt

    def model_parse(self):
        # gather options
        self.gather_options()
        if self.opt.mode == 'train' and not self.opt.resume:
            self.construct_checkpoints(creatDir = True)
        elif self.opt.mode == 'train' and self.opt.resume or self.opt.mode == 'test':
            self.construct_checkpoints(creatDir = False)
        self.construct_splitDir()
        self.construct_actionDir()

        # continue to train
        if self.opt.mode == 'train' and self.opt.resume:
            self.load_options('opt.txt')

        # print options
        self.construct_message()
        if self.opt.mode == 'train' and not self.opt.resume:
            self.save_options('opt.txt')
        if self.opt.mode == 'test':
            self.save_options('test_opt.txt')
        self.print_options()

        # set gpu ids
        self.construct_device()

        return self.opt

    def update(self):
        self.construct_message()
        self.save_options('opt.txt')
