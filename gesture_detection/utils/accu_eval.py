
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

    def compute_hist(self, pred, gnd):
        hist = torch.bincount((((self.nClass + 1) * gnd) + pred),minlength = (self.nClass + 1)**2)
        hist = hist.view(self.nClass + 1, self.nClass + 1)[:-1,:-1]
        return hist

    def update(self, pred, gnd):
        assert( pred.size() == gnd.size())
        self.area += pred.numel()
        self.intersection += torch.sum(pred.eq(gnd))

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
        print('\r{}'.format(i), end = '')
    print('elapsed_time: {}'.format(time.time() - start_time))
