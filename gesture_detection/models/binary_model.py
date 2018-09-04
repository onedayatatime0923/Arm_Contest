import itertools
import torch
from torch.autograd import Variable
from utils import AccuEval
from models.network.network import Classifier
from models.base_model import BaseModel, getOptimizer


class BinaryModel(BaseModel):
    def __init__(self, opt):
        super(BinaryModel, self).__init__(opt)
        print('-------------- Networks initializing -------------')

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.lossNames = ['loss{}'.format(i) for i in 
            ['Classifier']]
        self.lossClassifier = 0
        # specify the training miou you want to print out. The program will call base_model.get_current_losses

        self.accuNames = ['accu{}'.format(i) for i in 
            ['Classifier']]
        self.accuClassifier = AccuEval()

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        # naming is by the input domain
        self.modelNames = ['net{}'.format(i) for i in 
                ['Classifier']]

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_RGB (G), G_D (F), D_RGB (D_Y), D_D (D_X)
        self.netClassifier = self.initNet(Classifier(opt))

        self.set_requires_grad([self.netClassifier],True)

        # define loss functions
        self.criterion = torch.nn.BCELoss()
        # initialize optimizers
        self.optimizerC = getOptimizer(
            itertools.chain(self.netClassifier.parameters()),
            opt = opt.opt, lr=opt.lr, beta1 = opt.beta1,
            momentum = opt.momentum, weight_decay = opt.weight_decay)
        self.optimizers = []
        self.optimizers.append(self.optimizerC)


        print('--------------------------------------------------')
    def name(self):
        return 'BinaryModel'

    def set_input(self, input):
        self.signal = input['signal'].to(self.opt.device)
        self.label = input['label'].to(self.opt.device)

    def forward(self, data):
        data = Variable(data)
        output = self.netClassifier(data)
        return  float(output) > 0.5

    def backward_classifier(self, retain_graph = False):
        output = self.netClassifier(self.signal)

        pred = output>0.5
        self.accuClassifier.update(pred, self.label.byte())

        lossClassifier = self.criterion(output, self.label)
        lossClassifier.backward( retain_graph = retain_graph)
        self.lossClassifier = float(lossClassifier)

    def optimize_parameters(self):
        # update F and C for Source
        self.set_requires_grad([self.netClassifier], True)
        self.optimizerC.zero_grad()
        self.backward_classifier(retain_graph = False)
        self.optimizerC.step()
