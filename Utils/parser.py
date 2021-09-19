import argparse
import os

import numpy as np
import torch
import random

import re
import yaml

import shutil

from datetime import datetime


class Namespace(object):
    """
    load configs from xxx.yaml
    """
    def __init__(self, Dict):
        for key, value in Dict.items():
            assert isinstance(key, str) and re.match("[A-Za-z_-]", key)
            if isinstance(value, dict):
                self.__dict__[key] = Namespace(value)
            else:
                self.__dict__[key] = value

    def __getattr__(self, attribute):
        raise AttributeError(f"Can not find {attribute} in namespace. Please write {attribute} in config file!")


def set_deterministic(seed):
    """default is None"""
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', type=str, default="./configs/simsiam_cifar.yaml", help="xxx.yaml")
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--debug_subset_size', type=int, default=8)
    parser.add_argument('--download', action='store_true', help="if cannot fine dataset, download from web")
    parser.add_argument('--data_dir', type=str, default="/Users/luoyongjia/Research/Data/cifar10")
    parser.add_argument('--log_dir', type=str, default="./res/logs")
    parser.add_argument('--checkpoint_dir', type=str, default="./res/checkpoints")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--hide_progress', action='store_true')

    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        for key, value in Namespace(yaml.load(f, Loader=yaml.FullLoader)).__dict__.items():
            vars(args)[key] = value

    if args.debug:
        if args.train:
            args.train.batch_size = 2
            args.train.num_epochs = 1
            args.train.stop_at_epoch = 1
        if args.eval:
            args.eval.batch_size = 2
            args.eval.num_epochs = 1
        args.dataset.num_workers = 0

    assert not None in [args.log_dir, args.data_dir, args.name]

    os.makedirs(args.log_dir, exist_ok=False)
    print(f'Creating file {args.log_dir}')

    set_deterministic(args.seed)

    vars(args)['aug_kwargs'] = {
        'image_size': args.dataset.image_size,
    }
    vars(args)['dataset_kwargs'] = {
        'dataset': args.dataset.name,
        'data_dir': args.data_dir,
        'download': args.download,
        'debug_subset_size': args.debug_subset_size if args.debug else None,
    }
    vars(args)['dataloader_kwargs'] = {
        'drop_last': True,
        'pin_memory': True,
        'num_workers': args.dataset.num_workers,
    }

    return args


if __name__ == "__main__":
    args = get_args()
