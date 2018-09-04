
import os
import torch
import numpy as np
from dataset.base_dataset import BaseDataset


class BinaryDataSet(BaseDataset):
    def __init__(self, opt):
        self.opt = opt

        self.files = os.listdir(self.opt.dataDir)

        signal = []
        label = []
        for f in self.files:
            s, l = (np.load(os.path.join(self.dataDir, f)))
            signal.append(s)
            label.append(l)
        self.signal = np.concatenate(signal,0)
        self.label = np.concatenate(label,0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        signal = torch.FloatTensor(self.signal[index]).squeeze(1)
        label = torch.FloatTensor([self.label[index]])


        data={'signal': signal,
                'label': label}
        return data

    def name(self):
        return 'BinaryDataSet'
