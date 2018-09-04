

from dataset.binary_dataset import BinaryDataSet
from dataset.concat_dataset import ConcatDataset

assert BinaryDataSet and ConcatDataset

def createDataset(opt, split):
    return BinaryDataSet(opt.dataDir, split)

