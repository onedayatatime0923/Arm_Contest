
import sys
sys.path.append('./')
from options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    def __init__(self):
        BaseOptions.__init__(self)
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # ---------- Define Mode ---------- #
        parser.set_defaults(mode = 'train')
        # ---------- Define Dataset ---------- #
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
        parser.add_argument('--batchSize', type=int, default=2,
                            help="batch_size")
        parser.add_argument('--epoch', type=int, default=1,
                            help='the training epoch.')
        parser.add_argument('--nEpochStart', type=int, default=100,
                            help='# of epoch at starting learning rate')
        parser.add_argument('--nEpochDecay', type=int, default=100,
                            help='# of epoch to linearly decay learning rate to zero')
        # ---------- Experiment Setting ---------- #
        parser.set_defaults(name= 'train')
        parser.add_argument('--displayInterval', type=int, default=5,
                            help='frequency of showing training results on screen')
        parser.add_argument('--saveLatestInterval', type=int, default=5000,
                            help='frequency of saving the latest results')
        parser.add_argument('--saveEpochInterval', type=int, default=5, 
                            help='frequency of saving checkpoints at the end of epochs')
        return parser
    def parse(self):
        BaseOptions.model_parse(self)

        return self.opt
