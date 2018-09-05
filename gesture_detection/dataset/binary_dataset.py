
import os
import torch
import numpy as np
from dataset.base_dataset import BaseDataset


class BinaryDataSet(BaseDataset):
    def __init__(self, dataDir, split):
        self.dataDir = dataDir

        self.files = os.listdir(os.path.join(dataDir, split))
        print('reading from {}...'.format(' '.join(self.files)))
        assert( len(self.files) > 0)

        signal = []
        label = []
        for f in self.files:
            s, l = (np.load(os.path.join(dataDir, split, f)))
            signal.append(s)
            label.append(l)
        self.signal = np.concatenate(signal,0)
        self.label = np.concatenate(label,0)
        assert( len(self.signal) == len(self.label))

    def __len__(self):
        return len(self.signal)

    def __getitem__(self, index):
        signal = torch.FloatTensor(self.signal[index]).squeeze(1)
        label = torch.FloatTensor([self.label[index]])


        data={'signal': signal,
                'label': label}
        return data

    def name(self):
        return 'BinaryDataSet'
