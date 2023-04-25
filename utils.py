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

cmap = plt.cm.jet


def save_depth_as_uint8png(img, filename):
    image_to_write = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    image_to_write = image_to_write.astype('uint8')
    cv2.imwrite(filename, image_to_write)


def Uncertainty_nomalize(depth):
    depth = (depth - np.min(depth[100:,:])) / (np.max(depth[100:,:]) - np.min(depth[100:,:]))
    depth = 255 * cmap(depth)[:, :, :3]  # H, W, C
    return depth.astype('uint8')