import os
import os.path
import glob
import numpy as np
from random import choice
from PIL import Image
import torch.utils.data as data
import cv2
from dataloaders import transforms

def get_paths_and_transform(split, args):

    if split == "train":
        transform = train_transform
        glob_d = os.path.join(
            args.data_folder,
            'data_depth_velodyne/train/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png'
        )
        glob_gt = os.path.join(
            args.data_folder,
            'data_depth_annotated/train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
        )

        def get_rgb_paths(p):
            ps = p.split('/')
            pnew = '/'.join([args.data_folder] + ['data_rgb'] + ps[-6:-4] +
                            ps[-2:-1] + ['data'] + ps[-1:])
            return pnew

    elif split == "val":
        if args.val == "full":
            transform = val_transform
            glob_d = os.path.join(
                args.data_folder,
                'data_depth_velodyne/val/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png'
            )
            glob_gt = os.path.join(
                args.data_folder,
                'data_depth_annotated/val/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
            )
            def get_rgb_paths(p):
                ps = p.split('/')
                pnew = '/'.join(ps[:-7] +  
                    ['data_rgb']+ps[-6:-4]+ps[-2:-1]+['data']+ps[-1:])
                return pnew

        elif args.val == "select":
            transform = no_transform
            glob_d = os.path.join(
                args.data_folder,
                "depth_selection/val_selection_cropped/velodyne_raw/*.png")
            glob_gt = os.path.join(
                args.data_folder,
                "depth_selection/val_selection_cropped/groundtruth_depth/*.png"
            )
            def get_rgb_paths(p):
                return p.replace("groundtruth_depth","image")

    elif split == "test_completion":
        transform = no_transform
        glob_d = os.path.join(
            args.data_folder,
            "depth_selection/test_depth_completion_anonymous/velodyne_raw/*.png"
        )
        glob_gt = None  #"test_depth_completion_anonymous/"
        glob_rgb = os.path.join(
            args.data_folder,
            "depth_selection/test_depth_completion_anonymous/image/*.png")

    elif split == "test_prediction":
        transform = no_transform
        glob_d = None
        glob_gt = None  #"test_depth_completion_anonymous/"
        glob_rgb = os.path.join(
            args.data_folder,
            "depth_selection/test_depth_prediction_anonymous/image/*.png")
    else:
        raise ValueError("Unrecognized split " + str(split))

    if glob_gt is not None:
        # train or val-full or val-select
        paths_d = sorted(glob.glob(glob_d)) 
        paths_gt = sorted(glob.glob(glob_gt)) 
        paths_rgb = [get_rgb_paths(p) for p in paths_gt]
    else:  
        # test only has d or rgb
        paths_rgb = sorted(glob.glob(glob_rgb))
        paths_gt = [None] * len(paths_rgb)
        if split == "test_prediction":
            paths_d = [None] * len(
                paths_rgb)  # test_prediction has no sparse depth
        else:
            paths_d = sorted(glob.glob(glob_d))

    paths = {"rgb": paths_rgb, "d": paths_d, "gt": paths_gt}
    return paths, transform


def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    rgb_png = np.array(img_file, dtype='uint8')  # in the range [0,255]
    img_file.close()
    return rgb_png


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png),filename)

    depth = depth_png.astype(np.float) / 256.
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth, -1)
    return depth


oheight, owidth = 352, 1216


def drop_depth_measurements(depth, prob_keep):
    mask = np.random.binomial(1, prob_keep, depth.shape)
    depth *= mask
    return depth


def train_transform(rgb, sparse, target, args):
    # s = np.random.uniform(1.0, 1.5) # random scaling
    # angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
    do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

    transform_geometric = transforms.Compose([
        transforms.BottomCrop((oheight, owidth)),
        transforms.HorizontalFlip(do_flip)
    ])
    if sparse is not None:
        sparse = transform_geometric(sparse)
    target = transform_geometric(target)
    if rgb is not None:
        transform_rgb = transforms.Compose([
            transform_geometric
        ])
        rgb = transform_rgb(rgb)

    # sparse = drop_depth_measurements(sparse, 0.9)

    return rgb, sparse, target


def val_transform(rgb, sparse, target,args):
    transform = transforms.Compose([
        transforms.BottomCrop((oheight, owidth)),
    ])
    if rgb is not None:
        rgb = transform(rgb)
    if sparse is not None:
        sparse = transform(sparse)
    if target is not None:
        target = transform(target)
    return rgb, sparse, target


def no_transform(rgb, sparse, target,  args):
    return rgb, sparse, target


to_tensor = transforms.ToTensor()
to_float_tensor = lambda x: to_tensor(x).float()


def handle_gray(rgb):

    #img = cv2.resize(rgb, (608, 176), interpolation=cv2.INTER_CUBIC)
    img = np.array(Image.fromarray(rgb).convert('L'))
    img = np.expand_dims(img, -1)
    return rgb, img


class KittiDepth(data.Dataset):
    """A data loader for the Kitti dataset
    """
    def __init__(self, split, args):
        self.args = args
        self.split = split
        paths, transform = get_paths_and_transform(split, args)
        self.paths = paths
        self.transform = transform
        self.threshold_translation = 0.1

    def __getraw__(self, index):
        rgb = rgb_read(self.paths['rgb'][index])
        sparse = depth_read(self.paths['d'][index])
        if self.split == "test_completion" or self.split == "test_prediction":
            return rgb, sparse, None
        else:
            target = depth_read(self.paths['gt'][index])
            return rgb, sparse, target

    def __getitem__(self, index):
        rgb, sparse, target = self.__getraw__(index)
        rgb, sparse, target = self.transform(rgb, sparse, target, self.args)

        rgb, gray = handle_gray(rgb)
        rgb_HSV = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        h = np.shape(rgb)[0]
        w = np.shape(rgb)[1]
        shrink_8 = cv2.resize(rgb, (w//8,h//8), interpolation=cv2.INTER_CUBIC)
        shrink_4 = cv2.resize(rgb, (w // 4, h // 4), interpolation=cv2.INTER_CUBIC)
        shrink_2 = cv2.resize(rgb, (w // 2, h // 2), interpolation=cv2.INTER_CUBIC)
        shrink_1 = rgb

        hsv_8 = cv2.resize(rgb_HSV, (w//8,h//8), interpolation=cv2.INTER_CUBIC)
        hsv_4 = cv2.resize(rgb_HSV, (w // 4, h // 4), interpolation=cv2.INTER_CUBIC)
        hsv_2 = cv2.resize(rgb_HSV, (w // 2, h // 2), interpolation=cv2.INTER_CUBIC)
        hsv_1 = rgb_HSV
        #print("hsv_8",hsv_8.shape)
        gray_8 = cv2.resize(gray, (w//8,h//8), interpolation=cv2.INTER_CUBIC)
        gray_8 = np.expand_dims(gray_8, -1)
        gray_4 = cv2.resize(gray, (w // 4, h // 4), interpolation=cv2.INTER_CUBIC)
        gray_4 = np.expand_dims(gray_4, -1)
        gray_2 = cv2.resize(gray, (w // 2, h // 2), interpolation=cv2.INTER_CUBIC)
        gray_2 = np.expand_dims(gray_2, -1)
        gray_1 = gray
        #print("gray_8",gray_8.shape)
        candidates = {"rgb":rgb, "d":sparse, "gt":target, \
                      "gray_1":gray_1,"gray_2":gray_2,"gray_4":gray_4, "gray_8":gray_8,\
                      "rgb_8":shrink_8,"rgb_4":shrink_4,"rgb_2":shrink_2,"rgb_1":shrink_1,\
                      "hsv_8":hsv_8,"hsv_4":hsv_4,"hsv_2":hsv_2,"hsv_1":hsv_1}
        items = {
            key: to_float_tensor(val)
            for key, val in candidates.items() if val is not None
        }

        return items

    def __len__(self):
        return len(self.paths['gt'])
