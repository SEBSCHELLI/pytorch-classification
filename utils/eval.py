from __future__ import print_function, absolute_import

__all__ = ['accuracy']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    print(output.shape)
    print(target.shape)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    print(correct.shape)

    """import torch
    a = torch.Tensor([[1,2,3,4],
                      [2,3,2,3],
                      [2,3,2,3],
                      [2,3,2,3],
                      [2,3,2,3]])

    print(a.shape)"""

    res = []
    for k in topk:
        print(k)
        print(correct[:k].shape)
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res