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

import _init_paths
# import lib.models
# import lib.datasets
# from lib.config import config
# from lib.config import update_config
# from lib.utils.utils import create_logger, FullModel
# from lib.core.criterion import CrossEntropy

import models
import datasets
from config import config
from config import update_config
from tqdm import tqdm

def main():
    num_class = 20
    dataset_real_path = 'F:/Datasets/SinglePerson/final/'
    dataset_syn_path = 'F:/Datasets/genBySd/'
    filter_masks_path = 'F:/Datasets/genBySd/filter_masks'
    if not os.path.exists(filter_masks_path):
        os.makedirs(filter_masks_path)

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

    args.cfg = 'D:/linzhy/PycharmProjects/HRNet-for-dachuang/experiments/lip/seg_hrnet_ocr_w48_473x473_sgd_lr7e-3_wd5e-4_bs_40_epoch150_for_filter.yaml'

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

    dataset_real = datasets.lip(
        root=dataset_real_path,       # the path where the real dataset is located
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

    dataset_syn = datasets.lip(
        root=dataset_syn_path,       # the path where the synthetic dataset is located
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

    criterion = nn.CrossEntropyLoss(
        # weight=dataset_real.class_weights,
        ignore_index=config.TRAIN.IGNORE_LABEL,
        reduction='none'
    )
    # 不要使用这个
    # criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
    #                          weight=dataset_real.class_weights,
    #                          reduction='none')
    # 这里就是将前向传播和计算loss的操作整合到FullModel类中，只需调用forward方法即可。不过类中还提到整合分布在多个GPU上的损失，这里不是很明白
    # model = FullModel(model, criterion)
    gpus = list(config.GPUS)
    model = model.cuda()

    start = timeit.default_timer()

    # Calculate the class-wise mean loss on real images
    class_wise_mean_loss = [(0, 0) for _ in range(20)]

    with torch.no_grad():
        for index, batch in enumerate(tqdm(dataloader_real)):
            images, labels, _, _ = batch
            images = images.cuda()
            labels = labels.long().cuda()

            classes = torch.unique(labels).tolist()

            out_aux_seg = model(images)     # 网络的输出分为两部分，一部分应该是辅助计算loss的输出，一部分是最终的预测结果
            out_aux = out_aux_seg[0]
            outputs = out_aux_seg[1]
            # 一个神奇的现象是最终输出居然是1*20*119*119，而不是1*20*473*473??
            outputs = F.interpolate(input=outputs, size=(473, 473),
                                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

            losses = criterion(outputs, labels)
            # losses, _ = model(images, labels)
            label = labels.cpu().squeeze(dim=0)
            losses = losses.cpu().squeeze(dim=0)

            for class_ in classes:
                if class_ == 0:
                    continue
                pixel_num, loss_sum = class_wise_mean_loss[class_]
                class_wise_mean_loss[class_] = (pixel_num + torch.sum(label == class_).item(),
                                                loss_sum + torch.sum(losses[label == class_]).item())
    class_wise_mean_loss = [loss_sum / (pixel_num + 1e-5) for pixel_num, loss_sum in class_wise_mean_loss]
    print('Class-wise mean loss:', class_wise_mean_loss)

    # Filter out noisy synthetic pixels (criterion: loss > 1.25 * class-wise loss)
    with torch.no_grad():
        for index, batch in enumerate(tqdm(dataloader_syn)):
            images, labels, _, labels_name = batch
            images = images.cuda()
            labels = labels.long().cuda()

            classes = torch.unique(labels).tolist()

            out_aux_seg = model(images)
            out_aux = out_aux_seg[0]
            outputs = out_aux_seg[1]
            outputs = F.interpolate(input=outputs, size=(473, 473),
                                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

            losses = criterion(outputs, labels)
            # losses, _ = model(images, labels)
            label = labels.cpu().squeeze(dim=0)
            losses = losses.cpu().squeeze(dim=0)

            mask_filtered = torch.zeros_like(label)
            mask_filtered[:] = label[:]

            for class_ in classes:
                if class_ == 0:
                    continue
                # 容忍度设为1.25
                filtered_region = (label == class_) & (losses > class_wise_mean_loss[class_] * 1.25)
                mask_filtered[filtered_region] = 0
            mask_filtered = mask_filtered.numpy().astype(np.uint8)
            mask_filtered = Image.fromarray(mask_filtered)
            mask_filtered.save(os.path.join('F:/Datasets/genBySd/filter_masks', labels_name[0]+'.png'))

    end = timeit.default_timer()
    print('Mins: %d' % np.int_((end - start) / 60))
    print('Done')


if __name__ == '__main__':
    main()
