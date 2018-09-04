

# PyTorch includes
import torch
from torch.autograd import Variable
# Custom includes
from options import TrainOptions
from dataset import createDataset
from models import createModel
from visualizer import TrainVisualizer

assert Variable


parser = TrainOptions()
opt = parser.parse()

# set dataloader
dataset = createDataset(opt)

dataLoader= torch.utils.data.DataLoader(
    dataset,
    batch_size= opt.batchSize, shuffle=True,
    num_workers=opt.nThreads)

# set model

model = createModel(opt)
model.setup(opt)
# set visualizer

visualizer = TrainVisualizer(opt, dataLoader.dataset).reset()

steps = 0
for epoch in range(opt.epoch, opt.nEpochStart + opt.nEpochDecay + 1):
    for i, data in enumerate(dataLoader):
        steps += 1

        model.set_input(data)
        model.optimize_parameters()

        visualizer('Train', epoch, data = {**model.current_losses(),
                **model.current_accus()})

        if steps % opt.displayInterval == 0:
            visualizer.displayScalor({**model.current_losses(),
                **model.current_accus()}, steps)

        if steps % opt.saveLatestInterval == 0:
            print('\nsaving the latest model (epoch %d, total_steps %d)' % (epoch, steps))
            model.save_networks('latest')


    if epoch % opt.saveEpochInterval == 0:
        print('\nsaving the model at the end of epoch %d, iters %d' % (epoch, steps))
        model.save_networks('latest')
        model.save_networks(epoch)

    visualizer.end('Train', epoch)
    print('='*80)
    if opt.adjustLr:
        model.update_learning_rate()
