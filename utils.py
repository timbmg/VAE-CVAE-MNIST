import torch
from torch.autograd import Variable

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def idx2onehot(idx, n):

    assert idx.size(1) == 1
    assert torch.max(idx).data.item() < n

    onehot = torch.zeros(idx.size(0), n)
    if torch.cuda.is_available():
        onehot = onehot.cuda() #fix for error in idx.data being a cuda tensor
    onehot.scatter_(1, idx.data, 1)
    onehot = to_var(onehot)
    
    return onehot
