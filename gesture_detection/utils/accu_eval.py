
import torch


class AccuEval:
    def __init__(self, epsilon = 1E-8):
        self.epsilon = epsilon
        self.area = None
        self.intersection = None
        self.reset()

    def reset(self):
        self.area = 0
        self.intersection = 0
        return self

    def update(self, pred, gnd):
        assert( pred.size() == gnd.size())
        self.area += pred.numel()
        self.intersection += int(torch.sum(pred.eq(gnd)))

    def metric(self):
        accu = self.intersection / ( self.area + self.epsilon)

        return accu * 100
if __name__ == '__main__':
    import time
    n = 1000000
    c = 20
    length = 100
    start_time = time.time()
    for i in range(length):
        iouEval = AccuEval(c)
        a = torch.LongTensor(n).random_(0, c)
        b = torch.LongTensor(n).random_(0, c)
        iouEval.update(a,b)
        print('\r{}'.format(i), )
    print('elapsed_time: {}'.format(time.time() - start_time))
