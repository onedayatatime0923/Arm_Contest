

from dataset.binary_dataset import BinaryDataSet
from dataset.concat_dataset import ConcatDataset

assert BinaryDataSet and ConcatDataset

def createDataset(opt):
    return BinaryDataSet(opt.dataPath)

