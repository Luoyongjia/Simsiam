from tensorboardX import SummaryWriter

from torch import Tensor
from collections import OrderedDict
import os


def Logger(version, exp_num):
    def __init__(self, log_dir, tensorboard=True, matplotlib=True):

        self.reset(log_dir, tensorboard, matplotlib)

    def reset(self, log_dir=None, tensorboard=True, matplotlib=True):

        if log_dir is not None: self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir) if tensorboard else None
        self.counter = OrderedDict()

    def update_scalers(self, ordered_dict):

        for key, value in ordered_dict.items():
            if isinstance(value, Tensor):
                ordered_dict[key] = value.item()
            if self.counter.get(key) is None:
                self.counter[key] = 1
            else:
                self.counter[key] += 1

            if self.writer:
                self.writer.add_scalar(key, value, self.counter[key])


class Average_meter:
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.log = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.log.append(self.avg)
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
