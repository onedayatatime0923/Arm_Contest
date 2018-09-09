
import torch

def convert(data, n):
    data = data.view(1,n)
    return data

class Queue:
    def __init__(self, n, dim, device):
        self.n = n
        self.dim = dim
        self.device = device
        self.data = torch.zeros(n, dim).to(device)
    def __call__(self, data):
        self.data = torch.cat([self.data[1:,:],convert(data.to(self.device), self.dim)],0)
        return self.data
