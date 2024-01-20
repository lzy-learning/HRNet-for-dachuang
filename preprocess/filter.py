# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
import timeit
from pathlib import Path

import numpy as np

from PIL import Image

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import lib.models
import lib.datasets
from lib.config import config
from lib.config import update_config
from lib.utils.utils import create_logger, FullModel
from lib.core.criterion import CrossEntropy
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=False,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    args.cfg = './experiments/lip/seg_hrnet_ocr_w48_473x473_sgd_lr7e-3_wd5e-4_bs_40_epoch150.yaml'

    update_config(config, args)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    if torch.__version__.startswith('1'):
        module = eval('models.' + config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('models.' + config.MODEL.NAME + '.get_seg_model')(config)

    model_state_file = config.TEST.MODEL_FILE

    pretrained_dict = torch.load(model_state_file)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                       if k[6:] in model_dict.keys()}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])

    dataset_real = lib.datasets.lip(
        root='F:/Datasets/SinglePerson/final/',       # the path where the real dataset is located
        list_path='trainList.txt',
        num_samples=None,
        num_classes=20,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=473,
        crop_size=(473, 473),
        downsample_rate=1)
    dataloader_real = DataLoader(
        dataset_real,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    dataset_syn = lib.datasets.lip(
        root='F:/Datasets/genBySd/',       # the path where the synthetic dataset is located
        list_path='trainList.txt',
        num_samples=None,
        num_classes=20,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=473,
        crop_size=(473, 473),
        downsample_rate=1)
    dataloader_syn = DataLoader(
        dataset_syn,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                             weight=dataset_real.class_weights)
    # 这里就是将前向传播和计算loss的操作整合到FullModel类中，只需调用forward方法即可。不过类中还提到整合分布在多个GPU上的损失，这里不是很明白
    model = FullModel(model, criterion)
    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    start = timeit.default_timer()

    # Calculate the class-wise mean loss on real images
    class_wise_mean_loss = [(0, 0) for _ in range(21)]

    with torch.no_grad():
        for index, batch in enumerate(tqdm(dataloader_real)):
            images, labels, _, _ = batch
            images = images.cuda()
            labels = labels.long().cuda()

            classes = torch.unique(labels).tolist()

            losses, _ = model(images, labels)
            loss = losses.mean()        # 去除掉第0个维度，因为batch大小就是1了

            label = labels.cpu().squeeze(dim=0)

            for class_ in classes:
                if class_ == 0:
                    continue
                pixel_num, loss_sum = class_wise_mean_loss[class_]
                class_wise_mean_loss[class_] = (pixel_num + torch.sum(label == class_).item(),
                                                loss_sum + torch.sum(loss[label == class_]).item())
    class_wise_mean_loss = [loss_sum / (pixel_num + 1e-5) for pixel_num, loss_sum in class_wise_mean_loss]
    print('Class-wise mean loss:', class_wise_mean_loss)

    # Filter out noisy synthetic pixels (criterion: loss > 1.25 * class-wise loss)
    with torch.no_grad():
        for index, batch in enumerate(tqdm(dataloader_syn)):
            images, labels, _, labels_name = batch
            images = images.cuda()
            labels = labels.long().cuda()

            classes = torch.unique(labels).tolist()

            losses, _ = model(images, labels)
            loss = losses.mean()        # 去除掉第0个维度，因为batch大小就是1了

            label = labels.cpu().squeeze(dim=0)

            mask_filtered = torch.zeros_like(label)
            mask_filtered[:] = label[:]

            for class_ in classes:
                if class_ == 0:
                    continue
                # 容忍度设为1.25
                filtered_region = (label == class_) & (loss > class_wise_mean_loss[class_] * 1.25)
                mask_filtered[filtered_region] = 0
            mask_filtered = mask_filtered.numpy().astype(np.uint8)
            mask_filtered = Image.fromarray(mask_filtered)
            mask_filtered.save(os.path.join('F:/Datasets/genBySd/filter_masks', labels_name[0]+'.png'))

    end = timeit.default_timer()
    print('Mins: %d' % np.int_((end - start) / 60))
    print('Done')


if __name__ == '__main__':
    main()
