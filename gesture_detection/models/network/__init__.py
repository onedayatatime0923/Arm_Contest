
import torch
import torch.nn as nn

assert torch

class Classifier(nn.Module):
    def __init__(self, opt):
        super(Classifier, self).__init__()
        self.opt = opt
        self.feature = nn.Sequential(
            GaussianNoise(device = opt.device),
            nn.Linear(len(opt.input), opt.ncf),
            nn.BatchNorm1d(opt.ncf),
            nn.LeakyReLU(0.2, inplace=True),
            GaussianNoise(device = opt.device),
            nn.Dropout(opt.dropout),
            nn.Linear(opt.ncf, opt.ncf * 2),
            nn.BatchNorm1d(opt.ncf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            GaussianNoise(device = opt.device),
            nn.Dropout(opt.dropout),
            nn.Linear(opt.ncf * 2, opt.ncf * 4),
            nn.BatchNorm1d(opt.ncf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            GaussianNoise(device = opt.device),
            nn.Dropout(opt.dropout),
            nn.Linear(opt.ncf * 4, 1),
            nn.Sigmoid())
    def forward(self, x):
        x = torch.index_select(x, 1, torch.LongTensor(self.opt.input))
        x = self.feature(x)
        return x

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True, device = 0):
        nn.Module.__init__(self)
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.FloatTensor([0]).to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x 
