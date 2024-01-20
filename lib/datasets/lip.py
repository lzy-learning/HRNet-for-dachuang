# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np

import torch
from torch.nn import functional as F
from PIL import Image

from .base_dataset import BaseDataset


class LIP(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=20,
                 multi_scale=True,
                 flip=True,
                 ignore_label=-1,
                 base_size=473,
                 crop_size=(473, 473),
                 downsample_rate=1,
                 scale_factor=11,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        super(LIP, self).__init__(ignore_label, base_size,
                                  crop_size, downsample_rate, scale_factor, mean, std)

        self.root = root
        self.num_classes = num_classes
        self.list_path = list_path
        self.class_weights = None

        self.multi_scale = multi_scale
        self.flip = flip
        # strip()会默认去除换行符或空格，split默认以空格分割。注意这里直接用字符串加了，不知道为什么不用os.path.join
        self.img_list = [line.strip().split() for line in open(root + list_path)]
        # 返沪一个list，其中的元素对象包含训练图像路径、标签图像路径和图像名称
        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

    def read_files(self):
        files = []
        for item in self.img_list:
            if 'train' in self.list_path:
                image_path, label_path, _ = item
                # splittext函数默认分割文件后缀名
                name = os.path.splitext(os.path.basename(label_path))[0]
                sample = {"img": image_path,
                          "label": label_path,
                          "name": name, }
            elif 'val' in self.list_path:
                image_path, label_path, _ = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                sample = {"img": image_path,
                          "label": label_path,
                          "name": name, }
            else:
                raise NotImplementedError('Unknown subset.')
            files.append(sample)
        return files

    def resize_image(self, image, label, size):
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image_path = os.path.join(self.root, item['img'])
        label_path = os.path.join(self.root, item['label'])
        image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        )
        label = np.array(
            Image.open(label_path).convert('P')
        )

        size = label.shape
        if 'testval' in self.list_path:
            image = cv2.resize(image, self.crop_size,
                               interpolation=cv2.INTER_LINEAR)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), label.copy(), np.array(size), name

        if self.flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, ::flip, :]
            label = label[:, ::flip]

            if flip == -1:
                right_idx = [15, 17, 19]
                left_idx = [14, 16, 18]
                for i in range(0, 3):
                    right_pos = np.where(label == right_idx[i])
                    left_pos = np.where(label == left_idx[i])
                    label[right_pos[0], right_pos[1]] = left_idx[i]
                    label[left_pos[0], left_pos[1]] = right_idx[i]

        image, label = self.resize_image(image, label, self.crop_size)
        image, label = self.gen_sample(image, label,
                                       self.multi_scale, False)

        return image.copy(), label.copy(), np.array(size), name

    def inference(self, config, model, image, flip):
        size = image.size()
        pred = model(image)
        if config.MODEL.NUM_OUTPUTS > 1:
            pred = pred[config.TEST.OUTPUT_INDEX]

        pred = F.interpolate(
            input=pred, size=size[-2:],
            mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
        )

        if flip:
            flip_img = image.numpy()[:, :, :, ::-1]
            flip_output = model(torch.from_numpy(flip_img.copy()))

            if config.MODEL.NUM_OUTPUTS > 1:
                flip_output = flip_output[config.TEST.OUTPUT_INDEX]

            flip_output = F.interpolate(
                input=flip_output, size=size[-2:],
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )

            flip_output = flip_output.cpu()
            flip_pred = flip_output.cpu().numpy().copy()
            flip_pred[:, 14, :, :] = flip_output[:, 15, :, :]
            flip_pred[:, 15, :, :] = flip_output[:, 14, :, :]
            flip_pred[:, 16, :, :] = flip_output[:, 17, :, :]
            flip_pred[:, 17, :, :] = flip_output[:, 16, :, :]
            flip_pred[:, 18, :, :] = flip_output[:, 19, :, :]
            flip_pred[:, 19, :, :] = flip_output[:, 18, :, :]
            flip_pred = torch.from_numpy(
                flip_pred[:, :, :, ::-1].copy()).cuda()
            pred += flip_pred
            pred = pred * 0.5
        return pred.exp()

    def create_visual_anno(self, anno):
        # the corresponding rgb color of each label
        LIP_COLORMAP = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                                 [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                                 [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                                 [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                                 [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0]])
        # category corresponding to the label
        LIP_CLASS = ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses',
                     'UpperClothes', 'Dress', 'Coat', 'Socks', 'Pants', 'Jumpsuits',
                     'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                     'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']

        assert np.max(anno) <= 21

        # visualize
        visual_anno = np.zeros((anno.shape[0], anno.shape[1], 3), dtype=np.uint8)
        for i in range(visual_anno.shape[0]):  # i for h
            for j in range(visual_anno.shape[1]):
                color = LIP_COLORMAP[anno[i, j]]
                visual_anno[i, j, 0] = color[0]
                visual_anno[i, j, 1] = color[1]
                visual_anno[i, j, 2] = color[2]

        return visual_anno

    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            visual_anno = self.create_visual_anno(preds[i])
            save_img = Image.fromarray(visual_anno)
            save_img.save(os.path.join(sv_path, name[i] + '.jpg'))
