
import torch
import torch.nn as nn

assert torch

class Classifier(nn.Module):
    def __init__(self, opt):
        super(Classifier, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(opt.n, opt.ncf),
            nn.BatchNorm1d(opt.ncf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.dropout(opt.dropout),
            nn.Linear(opt.ncf, opt.ncf * 2),
            nn.BatchNorm1d(opt.ncf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.dropout(opt.dropout),
            nn.Linear(opt.ncf * 2, opt.ncf * 4),
            nn.BatchNorm1d(opt.ncf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.dropout(opt.dropout),
            nn.Linear(opt.ncf * 4, opt.ncf * 4),
            nn.BatchNorm1d(opt.ncf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.dropout(opt.dropout),
            nn.Linear(opt.ncf * 4, opt.ncf * 4),
            nn.BatchNorm1d(opt.ncf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.dropout(opt.dropout),
            nn.Linear(opt.ncf * 4, 1),
            nn.Sigmoid())
    def forward(self, x):
        x = self.feature(x)
        return x
