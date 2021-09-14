import random

import torch
import torchvision
from torchvision import transforms
from PIL import Image, ImageFilter

from Utils.parser import get_args

IMAGENETNORM = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]
CIFAR10NORM = [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]]


def get_dataset(dataset, data_dir, transform, train=True, download=False, debug_subset_size=None):
    if dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'imagenet':
        dataset = torchvision.datasets.ImageNet(data_dir, split='train' if train is True else 'val',
                                                transform=transform, download=download)
    else:
        raise NotImplementedError

    if debug_subset_size is not None:
        dataset = torch.utils.data.Subset(dataset, range(0, debug_subset_size))
        dataset.classes = dataset.dataset.classe
        dataset.targets = dataset.dataset.targets

    return dataset


class GaussianBlur(object):
    """from SimCLR."""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class transform_single:
    def __init__(self, image_size, train, normalize=CIFAR10NORM):
        if train is True:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*normalize)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(int(image_size*(8/7)), interpolation=Image.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(*normalize)
            ])

    def __call__(self, x):
        return self.transform(x)


class transform_simsiam:
    def __init__(self, image_size, normalize=CIFAR10NORM):
        image_size = 224 if image_size is None else image_size
        p_blur = 0.5 if image_size > 32 else 0
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(*normalize)
        ])

    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2


def get_transform(image_size=224, train=True, train_classifier=None):
    if train is True:
        augmentation = transform_simsiam(image_size)
    elif train is False:
        if train_classifier is None:
            raise Exception
        augmentation = transform_single(image_size, train = train_classifier)
    else:
        raise NotImplementedError

    return augmentation


def load_data(args):
    train_dataset = get_dataset(transform=get_transform(train=True, **args.aug_kwargs),
                                train=True,
                                **args.dataset_kwargs)
    memo_dataset = get_dataset(transform=get_transform(train=False, train_classifier=False, **args.aug_kwargs),
                               train=True,
                               **args.dataset_kwargs)
    test_dataset = get_dataset(transform=get_transform(train=False, train_classifier=False, **args.aug_kwargs),
                               train=False,
                               **args.dataset_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.train.batch_size,
                                               **args.dataloader_kwargs)
    memo_loader = torch.utils.data.DataLoader(memo_dataset, shuffle=False, batch_size=args.train.batch_size,
                                             **args.dataloader_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=256,
                                              **args.dataloader_kwargs)

    return train_loader, memo_loader, test_loader


if __name__ == "__main__":
    args = get_args()
    train_loader, _, _ = load_data(args)
