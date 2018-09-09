
import torch
import torch.nn as nn

assert torch

class Classifier(nn.Module):
    def __init__(self, opt):
        super(Classifier, self).__init__()
        self.opt = opt
        self.feature = nn.Sequential(
            nn.InstanceNorm1d(len(opt.input)),
            nn.Linear(len(opt.input), opt.ncf),
            nn.BatchNorm1d(opt.ncf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(opt.dropout),
            nn.Linear(opt.ncf, opt.ncf * 2),
            nn.BatchNorm1d(opt.ncf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(opt.dropout),
            nn.Linear(opt.ncf * 4, opt.ncf * 4),
            nn.BatchNorm1d(opt.ncf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(opt.dropout),
            nn.Linear(opt.ncf * 4, 1),
            nn.Sigmoid())
    def forward(self, x):
        x = torch.index_select(x, 1, torch.LongTensor(self.opt.input))
        x = self.feature(x)
        return x
