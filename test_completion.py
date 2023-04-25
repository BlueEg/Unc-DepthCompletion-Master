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
from model.model import MDCnet, AIRnet
from Utils.metrics import AverageMeter, Result
from Utils import criteria
from Utils import helper
import numpy as np
import matplotlib.pyplot as plt
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

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
                    default='',
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
cmap = plt.cm.jet

if cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("=> using '{}' for computation.".format(device))

# define loss functions
depth_criterion_L1 = criteria.UcertRELossL1()

depth_criterion = criteria.MaskedL1Loss()


def iterate(mode, loader, Refine_model, Base_model, logger, epoch):
    block_average_meter = AverageMeter()
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]
    # switch to appropriate mode
    assert mode == "test_completion", "unsupported mode: {}".format(mode)

    lr = 0

    for i, batch_data in enumerate(loader):
        start = time.time()
        batch_data = {key: val.to(device) for key, val in batch_data.items() if val is not None}
        gt = batch_data['gt'] if mode != 'test_prediction' and mode != 'test_completion' else None
        data_time = time.time() - start

        start = time.time()
        with torch.no_grad():
            pred_base, s_4 = Base_model(batch_data)
            res, pred = Refine_model(pred_base, batch_data)

        img = Uncertainty_nomalize(np.squeeze(res.data.cpu().numpy()))
        filename = os.path.join(logger.output_directory, '{0:010d}.png'.format(i))
        save_depth_as_uint8png(img, filename)

        gpu_time = time.time() - start

        # measure accuracy and record loss
        with torch.no_grad():
            mini_batch_size = next(iter(batch_data.values())).size(0)
            result = Result()

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

    Base_model = MDCnet().to(device)
    Bcheckpoint = torch.load("pretrained/model_best_step1.pth", map_location=device)
    Base_model.load_state_dict(Bcheckpoint['model'])
    Base_model = torch.nn.DataParallel(Base_model)
    Base_model.eval()

    Refine_model = AIRnet().to(device)
    Rcheckpoint = torch.load("pretrained/model_best_step2.pth", map_location=device)
    model_dict = Refine_model.state_dict()
    pretrained_dict = {k: v for k, v in Rcheckpoint['model'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    Refine_model.load_state_dict(model_dict)
    Refine_model = torch.nn.DataParallel(Refine_model)
    Refine_model.eval()

    # create backups and results folder
    logger = helper.logger(args, step='test_completion')

    test_dataset = KittiDepth('test_completion', args)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=2,
                                              pin_memory=True)
    print("\t==> test_loader size:{}".format(len(test_loader)))
    print("=> starting model evaluation ...")

    with torch.no_grad():
        result, is_best = iterate("test_completion", test_loader, Refine_model, Base_model, logger, Rcheckpoint['epoch'])

    return


if __name__ == '__main__':
    main()
