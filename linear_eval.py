import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from Utils import *
from models import *


def linear_eval(train_loader, test_loader, checkpoint, logger, args):
    model = get_backbone(args.model.backbone)
    classifier = nn.Linear(in_features=model.output_dim, out_features=10, bias=True).to(args.device)

    # assert args.eval_from is not None
    save_dict = checkpoint['state_dict']
    msg = model.load_state_dict({k[9:]: v for k, v in save_dict.items() if k.startswith('backbone.')},
                                strict=True)
    model = model.to(args.device)
    model = nn.DataParallel(model)

    # classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
    classifier = nn.DataParallel(classifier)

    optimizer = torch.optim.SGD(classifier.parameters(),
                                lr=args.eval.base_lr * args.eval.batch_size / 256,
                                momentum=args.eval.optimizer.momentum,
                                weight_decay=args.eval.optimizer.weight_decay)

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.eval.warmup_epochs, args.eval.warmup_lr,
        args.eval.num_epochs, args.eval.base_lr, args.eval.final_lr,
        len(train_loader),
    )

    loss_meter = Average_meter(name='Loss')
    acc_meter = Average_meter(name='Accuracy')

    # training
    global_progress = tqdm(range(0, args.eval.num_epochs), desc='Evaluating')
    for epoch in global_progress:
        loss_meter.reset()
        model.eval()
        classifier.train()
        local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{args.eval.num_epochs}', disable=True)

        for idx, (image, labels) in enumerate(local_progress):
            classifier.zero_grad()
            with torch.no_grad():
                feature = model(image.to(args.device))

            preds = classifier(feature)

            loss = F.cross_entropy(preds, labels.to(args.device))

            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            lr = lr_scheduler.step(epoch)
            local_progress.set_postfix({'lr': lr, 'loss': loss_meter.val, 'loss_avg': loss_meter.avg})

    classifier.eval()
    correct, total = 0, 0
    acc_meter.reset()
    for idx,  (image, labels) in enumerate(test_loader):
        with torch.no_grad():
            feature = model(image.to(args.device))
            preds = classifier(feature).argmax(dim=1)
            correct = (preds == labels.to(args.device)).sum().item()
            acc_meter.update(correct/preds.shape[0])
    logger.info(f'Accuracy = {acc_meter.avg * 100: .2f}')


if __name__ == "__main__":
    args = get_args()
