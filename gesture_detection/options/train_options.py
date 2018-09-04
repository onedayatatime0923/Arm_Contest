
import torch
import sys
sys.path.append('./')
from options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    def __init__(self):
        super(TrainOptions, self).__init__()
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # ---------- Define Mode ---------- #
        parser.add_argument('--mode', type=str, choices = ['train','test'], default = 'train',
                            help="Model Mode")
        # ---------- Define Dataset ---------- #
        parser.add_argument('--split', type=str, choices = ['train', 'val'],
                default = 'train')
        parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')
        # ---------- Optimizers ---------- #
        parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam',
                            help="network optimizer")
        parser.add_argument('--lr', type=float, default=1e-3,
                            help='learning rate (default: 0.001)')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--momentum', type=float, default=0.9,
                            help='momentum sgd (default: 0.9)')
        parser.add_argument('--weight_decay', type=float, default=2e-5,
                            help='weight_decay (default: 2e-5)')
        parser.add_argument("--adjustLr", action="store_true",
                            help='whether you change lr')
        parser.add_argument('--lr_policy', type=str, default='lambda',
                            help='learning rate policy: lambda|step|plateau')
        parser.add_argument('--lr_decay_iters', type=int, default=50,
                            help='multiply by a gamma every lr_decay_iters iterations')
        # ---------- Hyperparameters ---------- #
        parser.add_argument('--batchSize', type=int, default=1,
                            help="batch_size")
        parser.add_argument('--epoch', type=int, default=1,
                            help='the training epoch.')
        parser.add_argument('--nEpochStart', type=int, default=100,
                            help='# of epoch at starting learning rate')
        parser.add_argument('--nEpochDecay', type=int, default=100,
                            help='# of epoch to linearly decay learning rate to zero')
        # ---------- Whether to Resume ---------- #
        parser.add_argument("--resume", action = 'store_true',
                            help="whether to resume")
        parser.add_argument("--resumeName", type=str, default='latest',
                            help="model(pth) path, set to latest to use latest cached model")
        # ---------- Experiment Setting ---------- #
        parser.set_defaults(name= 'train')
        parser.add_argument('--verbose', action='store_true', 
                            help='if specified, print more debugging information')
        parser.add_argument('--displayInterval', type=int, default=5,
                            help='frequency of showing training results on screen')
        parser.add_argument('--saveLatestInterval', type=int, default=5000,
                            help='frequency of saving the latest results')
        parser.add_argument('--saveEpochInterval', type=int, default=5, 
                            help='frequency of saving checkpoints at the end of epochs')
        return parser
    def parse(self):
        # gather options
        self.gather_options()
        if self.opt.mode == 'train' and not self.opt.resume:
            self.construct_checkpoints(creatDir = True)
        elif self.opt.mode == 'train' and self.opt.resume:
            self.construct_checkpoints(creatDir = False)
        elif self.opt.mode == 'test':
            self.construct_checkpoints(creatDir = False)
            self.construct_outputPath()

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
    def construct_device(self):
        # set gpu ids
        if self.opt.gpuIds[0] != -1:
            self.opt.device = torch.device(self.opt.gpuIds[0])
        else:
            self.opt.device = torch.device('cpu')
