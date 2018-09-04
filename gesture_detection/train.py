

# PyTorch includes
import torch
from torch.autograd import Variable
# Custom includes
from options import TrainOptions
from dataset import createDataset
from models import createModel
from visualizer import TrainVisualizer
import errno

assert Variable


parser = TrainOptions()
opt = parser.parse()

# set dataloader
trainDataset = createDataset(opt, split = 'train')
valDataset = createDataset(opt, split = 'val')

trainDataLoader= torch.utils.data.DataLoader(
    trainDataset,
    batch_size= opt.batchSize, shuffle=True,
    num_workers=opt.nThreads)

valDataLoader= torch.utils.data.DataLoader(
    trainDataset,
    batch_size= opt.batchSize, shuffle=True,
    num_workers=opt.nThreads)
# set model

model = createModel(opt)
model.setup(opt)
# set visualizer

visualizer = TrainVisualizer(opt, trainDataLoader.dataset).reset()

steps = 0
for epoch in range(opt.epoch, opt.nEpochStart + opt.nEpochDecay + 1):
    # train
    for i, data in enumerate(trainDataLoader):
        steps += 1

        model.set_input(data)
        model.optimize_parameters()

        visualizer('Train', epoch, data = model.current_losses())

        if steps % opt.displayInterval == 0:
            visualizer.displayScalor(model.current_losses(),steps)

        if steps % opt.saveLatestInterval == 0:
            print('\nsaving the latest model (epoch %d, total_steps %d)' % (epoch, steps))
            model.save_networks('latest')

    while True:
        try:
            visualizer.end('Train', epoch, data = model.current_accus())
        except IOError, e:
            if e.errno != errno.EINTR:
                raise
        else:
            break
    
    

    # val
    for i, data in enumerate(valDataLoader):
        steps += 1

        model.set_input(data)
        model.predict()

        visualizer('val', epoch, data = model.current_losses())

    while True:
        try:
            visualizer.end('Val', epoch, data = model.current_accus())
        except IOError, e:
            if e.errno != errno.EINTR:
                raise
        else:
            break


    if epoch % opt.saveEpochInterval == 0:
        print('\nsaving the model at the end of epoch %d, iters %d' % (epoch, steps))
        model.save_networks('latest')
        model.save_networks(epoch)

    print('='*80)
    if opt.adjustLr:
        model.update_learning_rate()
