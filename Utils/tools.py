# from tensorboardX import SummaryWriter
import logging
import os
import time

from torch import Tensor
from collections import OrderedDict


# class Logger(object):
#     def __init__(self, log_dir, tensorboard=True, matplotlib=True):
#
#         self.reset(log_dir, tensorboard, matplotlib)
#
#     def reset(self, log_dir=None, tensorboard=True, matplotlib=True):
#
#         if log_dir is not None: self.log_dir = log_dir
#         self.writer = SummaryWriter(log_dir=self.log_dir) if tensorboard else None
#         self.counter = OrderedDict()
#
#     def update_scalers(self, ordered_dict):
#
#         for key, value in ordered_dict.items():
#             if isinstance(value, Tensor):
#                 ordered_dict[key] = value.item()
#             if self.counter.get(key) is None:
#                 self.counter[key] = 1
#             else:
#                 self.counter[key] += 1
#
#             if self.writer:
#                 self.writer.add_scalar(key, value, self.counter[key])


def Logger(exp_num):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))[:-4]

    if not os.path.exists(f'./res/{exp_num}'):
        os.mkdir(f'./res/{exp_num}')
    if not os.path.exists(f'./res/{exp_num}/logs'):
        os.mkdir(f'./res/{exp_num}/logs')
    logPath = f'./res/{exp_num}/logs/'
    logName = logPath + rq + '.log'
    fh = logging.FileHandler(logName, mode='a')
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    return logger


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
