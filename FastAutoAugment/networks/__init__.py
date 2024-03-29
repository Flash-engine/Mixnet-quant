import torch

from torch import nn
from torch.nn import DataParallel
import torch.backends.cudnn as cudnn

from networks.mixnet_dsq import dsq_mixnet_m

from networks.mixnet import mixnet_m


def get_model(conf, num_class=10, data_parallel=True):
    name = conf['type']

    if name == 'mixnet_m':
        model = mixnet_m()
    elif name == 'dsq_mixnet_m':
        model = dsq_mixnet_m()
    else:
        raise NameError('no model named, %s' % name)
    if data_parallel:
        model = model.cuda()
        model = DataParallel(model)
    else:
        model = model.to(device)
    cudnn.benchmark = True
    return model


def num_class(dataset):
    return {
        'cifar100': 100,
    }[dataset]
