import itertools
import logging
import math
import os
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch import nn, optim

import numpy as np

from tqdm import tqdm
from theconf import Config as C, ConfigArgumentParser

from common import get_logger
from data import get_dataloaders
from lr_scheduler import  adjust_learning_rate_mixnet
from metrics import accuracy, Accumulator
from networks import get_model, num_class

from warmup_scheduler import GradualWarmupScheduler


logger = get_logger('MicroNet')
logger.setLevel(logging.INFO)

# label smooth 
class LabelSmoothSoftmaxCE(nn.Module):
    def __init__(self,
                 lb_pos=0.9,
                 lb_neg=0.005,
                 reduction='mean',
                 lb_ignore=255,
                 ):
        super(LabelSmoothSoftmaxCE, self).__init__()
        self.lb_pos = lb_pos
        self.lb_neg = lb_neg
        self.reduction = reduction
        self.lb_ignore = lb_ignore
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, logits, label):
        logs = self.log_softmax(logits)
        ignore = label.data.cpu() == self.lb_ignore
        n_valid = (ignore == 0).sum()
        label[ignore] = 0
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)
        label = self.lb_pos * lb_one_hot + self.lb_neg * (1-lb_one_hot)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        label[[a, torch.arange(label.size(1)), *b]] = 0

        if self.reduction == 'mean':
            loss = -torch.sum(torch.sum(logs*label, dim=1)) / n_valid
        elif self.reduction == 'none':
            loss = -torch.sum(logs*label, dim=1)
        return loss

# mixup data and criterion
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# run a epoch
def run_epoch(model, loader, loss_fn, optimizer, desc_default='', epoch=0, writer=None, verbose=1, scheduler=None, is_train=False):
    tqdm_disable = bool(os.environ.get('TASK_NAME', ''))
    if verbose:
        loader = tqdm(loader, disable=tqdm_disable)
        loader.set_description('[%s %04d/%04d]' % (desc_default, epoch, C.get()['epoch']))

    metrics = Accumulator()
    cnt = 0
    total_steps = len(loader)
    steps = 0
    for data, label in loader:
        steps += 1
        data, label = data.cuda(), label.cuda()

        if is_train:
            data, targets_a, targets_b, lam = mixup_data(data, label, use_cuda=True)
            data, targets_a, targets_b = map(Variable, (data, targets_a, targets_b))
            preds = model(data)
            loss = mixup_criterion(loss_fn, preds, targets_a, targets_b, lam)
        else:
            preds = model(data)
            loss = loss_fn(preds, label)
        if optimizer:
            optimizer.zero_grad()
        if optimizer:
            loss.backward()
            if getattr(optimizer, "synchronize", None):
                optimizer.synchronize()
            if C.get()['optimizer'].get('clip', 5) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), C.get()['optimizer'].get('clip', 5))
            optimizer.step()

        top1, top5 = accuracy(preds, label, (1, 5))

        metrics.add_dict({
            'loss': loss.item() * len(data),
            'top1': top1.item() * len(data),
            'top5': top5.item() * len(data),
        })
        cnt += len(data)
        if verbose:
            postfix = metrics / cnt
            if optimizer:
                postfix['lr'] = optimizer.param_groups[0]['lr']
            loader.set_postfix(postfix)

        del preds, loss, top1, top5, data, label

    if tqdm_disable:
        logger.info('[%s %03d/%03d] %s', desc_default, epoch, C.get()['epoch'], metrics / cnt)

    metrics /= cnt
    if optimizer:
        metrics.metrics['lr'] = optimizer.param_groups[0]['lr']
    if verbose:
        for key, value in metrics.items():
            writer.add_scalar(key, value, epoch)
    return metrics


def train_and_eval(tag, dataroot, test_ratio=0.0, cv_fold=0, reporter=None, metric='last', \
        save_path=None,pretrained=None, only_eval=False):

    if not reporter:
        reporter = lambda **kwargs: 0

    max_epoch = C.get()['epoch']
    trainsampler, trainloader, validloader, testloader_ = get_dataloaders(C.get()['dataset'], C.get()['batch'],\
            dataroot, test_ratio, split_idx = cv_fold)

    # create a model & an optimizer
    model = get_model(C.get()['model'], num_class(C.get()['dataset']),data_parallel=True)

    # criterion = nn.CrossEntropyLoss()
    criterion = LabelSmoothSoftmaxCE()
    if C.get()['optimizer']['type'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=C.get()['lr'],
            momentum=C.get()['optimizer'].get('momentum', 0.9),
            weight_decay=C.get()['optimizer']['decay'],
            nesterov=C.get()['optimizer']['nesterov']
        )
    else:
        raise ValueError('invalid optimizer type=%s' % C.get()['optimizer']['type'])

    
    is_master = True
    logger.debug('is_master=%s' % is_master)

   #set schedulers
    lr_scheduler_type = C.get()['lr_schedule'].get('type', 'cosine')
    if lr_scheduler_type == 'cosine':
        logger.debug('cosine learn decay.')
        t_max = C.get()['epoch']
        if C.get()['lr_schedule'].get('warmup', None):
            t_max -= C.get()['lr_schedule']['warmup']['epoch']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=0.)
    elif lr_scheduler_type == 'mixnet_l':
        scheduler = adjust_learning_rate_mixnet(optimizer)
    else:
        raise ValueError('invalid lr_schduler=%s' % lr_scheduler_type)

    if C.get()['lr_schedule'].get('warmup', None):
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=C.get()['lr_schedule']['warmup']['multiplier'],
            total_epoch=C.get()['lr_schedule']['warmup']['epoch'],
            after_scheduler=scheduler
        )

    from tensorboardX import SummaryWriter
    writers = [SummaryWriter(log_dir='./logs/%s/%s' % (tag, x)) for x in ['train', 'valid', 'test']]

    result = OrderedDict()
    epoch_start = 1

    ## load model for training or evaluation
    if save_path and os.path.exists(save_path):
        data = torch.load(save_path)
        if 'model' in data:
            new_state_dict = {}
            for k, v in data['model'].items():
                if  'module.' not in k:
                    new_state_dict['module.' + k] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
            optimizer.load_state_dict(data['optimizer'])
            logger.info('ckpt epoch@%d' % data['epoch'])
            if data['epoch'] < C.get()['epoch']:
                epoch_start = data['epoch']
            else:
                only_eval = True
            logger.info('epoch=%d' % data['epoch'])
        else:
            model.load_state_dict(data)
        del data
    elif pretrained:
        assert os.path.exists(pretrained)
        ckt=torch.load(pretrained)
        model_dict = model.state_dict()
        if 'model' in ckt:
            new_state_dict = {}
            for k, v in ckt['model'].items():
                if 'module.' not in k:
                    new_state_dict['module.' + k] = v
                else:
                    new_state_dict[k] = v
            model_dict.update(new_state_dict)
            model.load_state_dict(model_dict)
        else:
            model_dict.update(ckt)
            model.load_state_dict(model_dict)

    ##Evaluate the model
    if only_eval:
        print('Eval model')
        logger.info('evaluation only+')
        model.eval()
        rs = dict()
        
        rs['test'] = run_epoch(model, testloader_, criterion, None, desc_default='*test', epoch=0, writer=writers[2])

        for key, setname in itertools.product(['loss', 'top1', 'top5'], ['test']):
            result['%s_%s' % (key, setname)] = rs[setname][key]
        result['epoch'] = 0
        return result

    # train loop
    best_valid_loss = 10e10
    best_accuracy = 0.0
    for epoch in range(epoch_start, max_epoch + 1):
        scheduler.step()
        model.train()
        rs = dict()
        rs['train'] = run_epoch(model, trainloader, criterion, optimizer, desc_default='train', \
                epoch=epoch, writer=writers[0], verbose=is_master, scheduler=scheduler, is_train=True)
        model.eval()

        if math.isnan(rs['train']['loss']):
            raise Exception('train loss is NaN.')

        if (epoch % 1) == 0 or epoch == max_epoch:
            rs['valid'] = run_epoch(model, validloader, criterion, None, desc_default='valid', \
                    epoch=epoch, writer=writers[1], verbose=is_master)
            rs['test'] = run_epoch(model, testloader_, criterion, None, desc_default='*test', \
                    epoch=epoch, writer=writers[2], verbose=is_master)

            if metric == 'last' or rs[metric]['loss'] < best_valid_loss: 
                if metric != 'last':
                    best_valid_loss = rs[metric]['loss']
                for key, setname in itertools.product(['loss', 'top1', 'top5'], ['train', 'valid', 'test']):
                    result['%s_%s' % (key, setname)] = rs[setname][key]
                result['epoch'] = epoch

                writers[1].add_scalar('valid_top1/best', rs['valid']['top1'], epoch)
                writers[2].add_scalar('test_top1/best', rs['test']['top1'], epoch)

                reporter(
                    loss_valid=rs['valid']['loss'], top1_valid=rs['valid']['top1'],
                    loss_test=rs['test']['loss'], top1_test=rs['test']['top1']
                )

            # save checkpoint
            if is_master and save_path:
                if rs['test']['top1'] > best_accuracy:
                    best_accuracy = rs['test']['top1']
                    logger.info('save model@%d to %s' % (epoch, save_path))
                    torch.save({
                        'epoch': epoch,
                        'log': {
                            'train': rs['train'].get_dict(),
                            'valid': rs['valid'].get_dict(),
                            'test': rs['test'].get_dict(),
                        },
                        'optimizer': optimizer.state_dict(),
                        'model': model.state_dict()
                    }, save_path)

    del model

    return result


if __name__ == '__main__':
    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--dataroot', type=str, default='', help='torchvision data folder')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--pretrained', type=str, default='')#loading pretrained model path
    parser.add_argument('--cv_ratio', type=float, default=0.0)
    parser.add_argument('--cv', type=int, default=0)
    parser.add_argument('--decay', type=float, default=-1)
    parser.add_argument('--only_eval', action='store_true')
    args = parser.parse_args()

    if args.decay > 0:
        logger.info('decay reset=%.8f' % args.decay)
        C.get()['optimizer']['decay'] = args.decay
    if args.save:
        logger.info('checkpoint will be saved at %s', args.save)

    ## for reproductivity
    import random
    random.seed(1)
    torch.manual_seed(1)
    cudnn.deterministic = True

    import time
    t = time.time()
    result = train_and_eval(args.tag, args.dataroot, \
            test_ratio=args.cv_ratio, cv_fold=args.cv, \
            save_path=args.save, pretrained=args.pretrained,\
            only_eval=args.only_eval)

    elapsed = time.time() - t

    logger.info('training done.')
    logger.info('model: %s' % C.get()['model'])
    logger.info(result)
    logger.info('elapsed time: %.3f Hours' % (elapsed / 3600.))
    logger.info('top1 error in testset: %.4f' % (1. - result['top1_test']))
