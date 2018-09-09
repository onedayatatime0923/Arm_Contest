
import sys
sys.path.append('./')
import os
import torch
import numpy as np
from dataset.base_dataset import BaseDataset


class BinaryDataSet(BaseDataset):
    def __init__(self, dataDir, split, nInput = 3):
        self.dataDir = dataDir
        self.nInput = nInput

        self.files = os.listdir(os.path.join(dataDir, split))
        print('reading from {}...'.format(' '.join(self.files)))
        assert( len(self.files) > 0)

        signal = []
        label = []
        for f in self.files:
            s, l = (np.load(os.path.join(dataDir, split, f)))
            s = np.array([ i for i in s])
            signal.append(s)
            label.append(l)
        self.signal = np.squeeze(np.concatenate(signal,0),2)
        self.label = np.concatenate(label,0)
        assert( len(self.signal) == len(self.label))

    def __len__(self):
        return len(self.signal) + 1 - self.nInput

    def __getitem__(self, index):
        signal = np.concatenate(self.signal[index:index + self.nInput])
        signal = torch.FloatTensor(self.signal[index:index + self.nInput]).view(-1)
        label = torch.FloatTensor([self.label[index -1 + self.nInput]])


        data={'signal': signal,
                'label': label}
        return data

    def name(self):
        return 'BinaryDataSet'
