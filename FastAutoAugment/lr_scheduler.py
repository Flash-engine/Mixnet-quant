import torch

from theconf import Config as C


def adjust_learning_rate_mixnet(optimizer):
    """
    Sets the learning rate to the initial LR decayed by 10 every [150, 225] epochs
    Ref: AutoAugment
    """

    return torch.optim.lr_scheduler.MultiStepLR(optimizer, [150, 225])
 
