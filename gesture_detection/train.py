

# PyTorch includes
import torch
from torch.autograd import Variable
# Custom includes
from options import TrainOptions
from dataset import createDataset
from models import createModel
from visualizer import ProgressVisualizer

assert Variable


parser = TrainOptions()
opt = parser.parse()

# set dataloader
trainDataset = createDataset(opt, split = 'train', nInput = opt.nInput)
valDataset = createDataset(opt, split = 'val', nInput = opt.nInput)

trainDataLoader= torch.utils.data.DataLoader(
    trainDataset,
    batch_size= opt.batchSize, shuffle=True,
    num_workers=opt.nThreads)

valDataLoader= torch.utils.data.DataLoader(
    valDataset,
    batch_size= opt.batchSize, shuffle=True,
    num_workers=opt.nThreads)
# set model

model = createModel(opt)
model.setup(opt)
# set visualizer

trainVisualizer = ProgressVisualizer(opt, trainDataLoader.dataset).reset()
valVisualizer = ProgressVisualizer(opt, valDataLoader.dataset).reset()

steps = 0
for epoch in range(opt.epoch, opt.nEpochStart + opt.nEpochDecay + 1):
    # train
    model.train()
    for i, data in enumerate(trainDataLoader):
        steps += 1

        model.set_input(data)
        model.optimize_parameters()

        trainVisualizer('Train', epoch, data = model.current_losses())
        if steps % opt.saveLatestInterval == 0:
            print('\nsaving the latest model (epoch %d, total_steps %d)' % (epoch, steps))
            model.save_networks('latest')

    trainVisualizer.end('Train', epoch, data = model.current_accus())
    

    # val
    model.eval()
    for i, data in enumerate(valDataLoader):
        steps += 1

        model.set_input(data)
        model.test()
        valVisualizer('Val', epoch, data = model.current_losses())
    valVisualizer.end('Val', epoch, data = model.current_accus())

    if epoch % opt.saveEpochInterval == 0:
        print('\nsaving the model at the end of epoch %d, iters %d' % (epoch, steps))
        model.save_networks('latest')
        model.save_networks(epoch)

    print('='*80)
    if opt.adjustLr:
        model.update_learning_rate()
