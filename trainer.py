import os
import torch
import torch.optim.optimizer as Op

from tqdm import tqdm
from Utils import *
from dataLoader import load_data
from models import Simsiam
from linear_eval import linear_eval


def save_checkpoint(model, optimizer, args, epoch):
    print('\nModel Saving...')
    if args.device_num > 1:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    torch.save({
        'model_state_dict': model_state_dict,
        'global_epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join('checkpoints', 'checkpoint_pretrain_model.pth'))


def train(train_loader, memo_loader, test_loader, model, optimizer, scheduler, args):
    model = model.to(args.device)
    model = torch.nn.DataParallel(model)

    logger = Logger('train', args.exp_num)
    accuracy = 0

    global_progress = tqdm(range(0, args.train.stop_at_epoch), desc='Training')
    for epoch in global_progress:
        model.eval()
        local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=True)
        for idx, ((image1, image2), labels) in enumerate(local_progress):
            model.zero_grad()
            data_dict = model.forward(image1.to(args.device, non_blocking=True), image2.to(args.device, non_blocking=True))
            loss = data_dict['loss'].mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            data_dict.update({'lr': scheduler.get_lr()})

            local_progress.set_postfix(data_dict)
            logger.log(data_dict)

        if args.train.knn_monitor and epoch % args.train.knn_interval == 0:
            accuracy = knn_monitor(model.module.backbone, memo_loader, test_loader, k=min(args.train.knn_k, len(memo_loader.dataset)), hideprogress=args.hide_progress)

        epoch_dict = {"epoch": epoch, "accuracy": accuracy}
        global_progress.set_postfix(epoch_dict)
        logger.update_scalers(epoch_dict)

        if epoch % args.savefrequency == 0:
            if not os.path.exists(f'./res/{args.exp_num}'):
                os.mkdir(f'./res/{args.exp_numo}')

            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict()
            }, f'./res/{args.exp_numo}/checkpoint-{epoch}.pth.tar')

    # save checkpoint
    torch.save({
        'epoch': args.train.stop_at_epoch,
        'state_dict': model.module.state_dict()
    }, f'./res/{args.exp_numo}/checkpoint-latest.pth.tar')
    print(f"Model saved to './res/{args.exp_numo}/checkpoint-latest.pth.tar'")


if __name__ == "__main__":
    args = get_args()
    # args.debug = True
    train_loader, memo_loader, test_loader = load_data(args)
    model = Simsiam(args)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.train.base_lr*args.train.batch_size/256,
                                momentum=args.train.optimizer.momentum,
                                weight_decay=args.train.optimizer.weight_decay)
    scheduler = LR_Scheduler(optimizer,
                             args.train.warmup_epochs, args.train.warmup_lr*args.train.batch_size/256,
                             args.train.num_epochs, args.train.base_lr*args.train.batch_size/256, args.train.final_lr*args.train.batch_size/256,
                             len(train_loader),
                             constant_predictor_lr=True)
    train(train_loader, memo_loader, test_loader, model, optimizer, scheduler, args)

    if args.eval is not False:
        args.checkpoint_dir = f'./res/{args.exp_numo}/checkpoint-latest.pth.tar'
        checkpoint = torch.load(args.checkpoint_dir)
        linear_eval(train_loader, test_loader, checkpoint, args)

    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')

    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')