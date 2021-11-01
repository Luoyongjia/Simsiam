import torch
import math
import numpy as np


class LR_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epoch, base_lr, final_lr, iter_per_epoch, constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epoch - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.init_lr = base_lr
        self.num_epoch = num_epoch
        self.iter = 0
        self.current_lr = 0

    def step(self, epoch):
        for param_group in self.optimizer.param_groups:
            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]

        self.iter += 1
        self.current_lr = lr
        return lr
    # def step(self, epoch):
    #     """Decay the learning rate based on schedule"""
    #     cur_lr = self.init_lr * 0.5 * (1. + math.cos(math.pi * epoch / self.num_epoch))
    #     for param_group in self.optimizer.param_groups:
    #         if 'fix_lr' in param_group and param_group['fix_lr']:
    #             param_group['lr'] = self.init_lr
    #         else:
    #             param_group['lr'] = cur_lr
    #
    #     self.current_lr = cur_lr
    #     return cur_lr

    def get_lr(self):
        return self.current_lr


