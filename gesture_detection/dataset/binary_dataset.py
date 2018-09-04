
import torch
import numpy as np
from dataset.base_dataset import BaseDataset


class BinaryDataSet(BaseDataset):
    def __init__(self, dataPath):
        self.dataPath = dataPath

        self.data = np.load(dataPath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        signal = torch.FloatTensor(self.data[0,index]).squeeze(1)
        label = torch.FloatTensor([self.data[1,index]])


        data={'signal': signal,
                'label': label}
        return data

    def name(self):
        return 'BinaryDataSet'
