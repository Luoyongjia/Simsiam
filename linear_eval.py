import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from Utils import *
from models import *


def linear_eval(train_loader, test_loader, checkpoint, args):
    model = get_backbone(args.model.backbone)
    classifier = nn.Linear(in_features=model.output_dim, out_features=10, bias=True).to(args.device)

    assert args.eval_from is not None
    save_dict = checkpoint['state_dict']
    msg = model.load_state_dict({k[9:]: v for k, v in save_dict['state_dict'].items() if k.startswith('backbone.')},
                                strict=True)
    model = model.to(args.device)
    model = nn.DataParallel(model)

    # classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
    classifier = nn.DataParallel(classifier)

    optimizer = torch.optim.SGD(classifier.parameters(),
                                lr=args.eval.base_lr * args.eval.batch_size / 256,
                                momentum=args.eval.optimizer.momentum,
                                weight_decay=args.eval.optimizer.weight_decay)



if __name__ == "__main__":
    args = get_args()