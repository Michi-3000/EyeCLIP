# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from show import *
class dfDataset(torch.utils.data.Dataset):
    def __init__(self, y='',transform=None,):
        df=mysearch('/home/danli/caption/multilabel/labelme_val_ming/')
        print(df.columns)
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self._consecutive_errors = 0
        self.y = y
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        row = self.df.iloc[index]       
        # target = self.df.loc[index,self.y]
        target=0
      
        img = Image.open(row['impath'])

        if self.transform is not None:
            img = self.transform(img)
        # print(target)
        # return {'image':img,'label':target}
        return img,target

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    #root = os.path.join(args.data_path, 'train' if is_train else 'val')
    #dataset = datasets.ImageFolder(root, transform=transform)
    df=mysearch('/home/danli/caption/multilabel/labelme_val_ming/')
    dataset = dfDataset(transform=transform)
    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
