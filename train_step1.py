import argparse
import os
import time
import torch.nn as nn
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import cv2
from dataloaders.kitti_loader import KittiDepth
from model.model import MDCnet
from Utils.metrics import AverageMeter, Result
from Utils import criteria
from Utils import helper
import numpy as np
import matplotlib.pyplot as plt
from utils import *

parser = argparse.ArgumentParser(description='Multi-scale_DepthCompletion')
parser.add_argument('-w',
                    '--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs',
                    default=20,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run (default: 11)')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-c',
                    '--criterion',
                    metavar='LOSS',
                    default='Ucertl2',
                    choices=criteria.loss_names,
                    help='loss function: | '.join(criteria.loss_names) +
                    ' (default: l2)')
parser.add_argument('-b',
                    '--batch-size',
                    default=2,
                    type=int,
                    help='mini-batch size (default: 1)')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=1e-4,
                    type=float,
                    metavar='LR',
                    help='initial learning rate (default 1e-5)')
parser.add_argument('--weight-decay',
                    '--wd',
                    default=0,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 0)')
parser.add_argument('--print-freq',
                    '-p',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume',
                    default=None,
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--data-folder',
                    default='/home/zhuyufan/Data/DepthCompletion/data',
                    type=str,
                    metavar='PATH',
                    help='data folder (default: none)')

parser.add_argument('--val',
                    type=str,
                    default="select",
                    choices=["select", "full"],
                    help='full or select validation set')
parser.add_argument('--jitter',
                    type=float,
                    default=0.1,
                    help='color jitter for images')
parser.add_argument(
    '--rank-metric',
    type=str,
    default='rmse',
    choices=[m for m in dir(Result()) if not m.startswith('_')],
    help='metrics for which best result is sbatch_datacted')

parser.add_argument('-e', '--evaluate', default='', type=str, metavar='PATH')

parser.add_argument('--cpu', action="store_true", help='run on cpu')

parser.add_argument('--save_pred',
                    type=bool,
                    default=True,
                    help='save pred depth')

args = parser.parse_args()

args.result = os.path.join('results')

print(args)

cuda = torch.cuda.is_available() and not args.cpu

if cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("=> using '{}' for computation.".format(device))

# define loss functions
depth_criterion_ucert = criteria.Ucertl2MaskedMSELoss()

depth_criterion = criteria.MaskedMSELoss()


def iterate(mode, args, loader, model, optimizer, logger, epoch):
    block_average_meter = AverageMeter()
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]
    pool = nn.MaxPool2d((2, 2))
    # switch to appropriate mode
    assert mode in ["train", "val"], "unsupported mode: {}".format(mode)

    if mode == 'train':
        model.train()
        lr = helper.adjust_learning_rate(args.lr, optimizer, epoch)
    else:
        model.eval()
        lr = 0

    for i, batch_data in enumerate(loader):

        start = time.time()
        batch_data = {key: val.to(device) for key, val in batch_data.items() if val is not None}
        gt = batch_data['gt']
        data_time = time.time() - start

        start = time.time()
        if mode == 'train':
            pred_1, pred_2, pred_3, pred_4, s_1, s_2, s_3, s_4 = model(batch_data)

        else:
            pred_4, s_4 = model(batch_data)

        gt4 = gt
        gt3 = pool(gt4)
        gt2 = pool(gt3)
        gt1 = pool(gt2)

        pred = pred_4
        if mode == 'train':
            # Loss 1: the direct depth supervision from ground truth label
            # mask=1 indicates that a pixel does not ground truth labels

            if epoch < 5:
                depth_loss = 0.45 * depth_criterion_ucert(pred_4, s_4, gt4) \
                             + 0.25 * depth_criterion_ucert(pred_3, s_3, gt3) \
                             + depth_criterion_ucert(pred_2, s_2, gt2) * 0.20 \
                             + depth_criterion_ucert(pred_1, s_1, gt1) * 0.10
            elif epoch < 10:
                depth_loss = 0.5 * depth_criterion_ucert(pred_4, s_4, gt4) \
                            + 0.20 * depth_criterion_ucert(pred_3,s_3,gt3) \
                            + depth_criterion_ucert(pred_2, s_2, gt2) * 0.20 \
                            + depth_criterion_ucert( pred_1,s_1, gt1) * 0.10
            else:
                depth_loss = depth_criterion_ucert(pred_4,s_4, gt)

            if epoch < 5:
                depth_losso = 0.45 * depth_criterion(pred_4, gt4) \
                              + 0.25 * depth_criterion(pred_3, gt3) \
                              + depth_criterion(pred_2, gt2) * 0.20 \
                              + depth_criterion(pred_1, gt1) * 0.10
            elif epoch < 10:
                depth_losso = 0.5 * depth_criterion(pred_4, gt4) \
                              + 0.20 * depth_criterion(pred_3, gt3) \
                              + depth_criterion(pred_2, gt2) * 0.20 \
                              + depth_criterion(pred_1, gt1) * 0.10
            else:
                depth_losso = depth_criterion(pred_4, gt)

            if (i + 1) % args.print_freq == 0:
                print(
                    'Epoch: {0} [{1}] :Loss with Uncertainty = {loss} , Orirginal Loss:{losso} '.format(epoch, i + 1,
                                                                                                        loss=depth_loss,
                                                                                                        losso=depth_losso))

            loss = depth_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        gpu_time = time.time() - start

        # measure accuracy and record loss
        with torch.no_grad():
            mini_batch_size = next(iter(batch_data.values())).size(0)
            result = Result()

            result.evaluate(pred.data, gt.data)
            [m.update(result, gpu_time, data_time, mini_batch_size) for m in meters]
            logger.conditional_print(mode, i, epoch, lr, len(loader), block_average_meter, average_meter)
            logger.conditional_save_img_comparison(mode, i, batch_data, pred, epoch)
            logger.conditional_save_pred(mode, i, pred)

    avg = logger.conditional_save_info(mode, average_meter, epoch)
    is_best = logger.rank_conditional_save_best(mode, avg, epoch)
    if is_best and not (mode == "train"):
        logger.save_img_comparison_as_best(mode, epoch)
    logger.conditional_summarize(mode, avg, is_best)

    return avg, is_best


def main():
    global args
    checkpoint = None

    if args.resume:  # optionally resume from a checkpoint
        args_new = args
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}' ... ".format(args.resume), end='')
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint['epoch'] + 1
            args.data_folder = args_new.data_folder
            args.val = args_new.val
            print("Completed. Resuming from epoch {}.".format(checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            return

    print("=> creating model and optimizer ... ", end='')
    model = MDCnet().to(device)
    model_named_params = [p for _, p in model.named_parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(model_named_params, lr=args.lr, weight_decay=args.weight_decay)

    if checkpoint is not None:
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> checkpoint state loaded.")

    model = torch.nn.DataParallel(model)

    train_dataset = KittiDepth('train', args)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=None)
    print("\t==> train_loader size:{}".format(len(train_loader)))
    val_dataset = KittiDepth('val', args)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True)  # set batch size to be 1 for validation
    print("\t==> val_loader size:{}".format(len(val_loader)))

    # create backups and results folder
    logger = helper.logger(args, step='step1')

    # main loop
    print("=> starting main loop ...")
    for epoch in range(args.start_epoch, args.epochs):
        print("=> starting training epoch {} ..".format(epoch))
        iterate("train", args, train_loader, model, optimizer, logger, epoch)

        checkpoint_filename = os.path.join(logger.output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
        torch.save({
            'epoch': epoch,
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': args}, checkpoint_filename)

        with torch.no_grad():
            result, is_best = iterate("val", args, val_loader, model, None, logger,
                                      epoch)  # evaluate on validation set
        helper.save_checkpoint({ # save checkpoint
            'epoch': epoch,
            'model': model.module.state_dict(),
            'best_result': logger.best_result,
            'optimizer': optimizer.state_dict(),
            'args': args,
        }, is_best, epoch, logger.output_directory)


if __name__ == '__main__':
    main()
