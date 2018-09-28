# -*- coding: utf-8 -*-
'''
Created by shinerio
reference @ author: herbert-chen
'''
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from model import model_v4
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
from config import config
from utils.logger import Logger
from utils.adjuster import adjustLR
import pprint
from data.TestDataset import TestDataset
from data.TrainDataset import TrainDataset
from inference import train, test, validate


def main():
    # 记录日志
    logger = Logger(config)
    logger.info('config:\n{}'.format(pprint.pformat(config)))
    # 随机种子
    seed = 666
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    # 数据增强：在给定角度中随机进行旋转
    class FixedRotation(object):
        def __init__(self, angles):
            self.angles = angles

        def __call__(self, img):
            return fixed_rotate(img, self.angles)

    def fixed_rotate(img, angles):
        angles = list(angles)
        angles_num = len(angles)
        index = random.randint(0, angles_num - 1)
        return img.rotate(angles[index])

    # 保存最新模型以及最优模型
    def save_checkpoint(state, filename):
        logger.info("save model to {}".format(filename))
        torch.save(state, filename)

    # 设定GPU ID
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    # 创建inception_v4模型
    model = model_v4.v4(num_classes=config.train.num_classes)
    model = torch.nn.DataParallel(model).cuda()

    # 读取训练图片列表
    all_data = pd.read_csv(config.train.label)
    # 读取测试图片列表
    test_data_list = pd.read_csv(config.test.imageList)
    # 图片归一化，由于采用ImageNet预训练网络，因此这里直接采用ImageNet网络的参数
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 分离训练集和测试集，stratify参数用于分层抽样
    if config.val_ratio is not None:
        train_data_list, val_data_list = train_test_split(all_data, test_size=config.val_ratio, random_state=666, stratify=all_data['label'])
        # 验证集图片变换
        val_data = TrainDataset(val_data_list,
                                transform=transforms.Compose([
                                    transforms.Resize((400, 400)),
                                    transforms.CenterCrop(384),
                                    transforms.ToTensor(),
                                    normalize,
                                ]))
        val_loader = DataLoader(val_data, batch_size=config.test.batch_size, shuffle=False, pin_memory=False,
                                num_workers=config.workers)
    else:
        train_data_list = all_data

    # 训练集图片变换，输入网络的尺寸为384*384
    train_data = TrainDataset(train_data_list,
                              transform=transforms.Compose([
                                  transforms.Resize((400, 400)),
                                  transforms.ColorJitter(0.15, 0.15, 0.15, 0.075),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomGrayscale(),
                                  # transforms.RandomRotation(20),
                                  FixedRotation([0, 90, 180, 270]),
                                  transforms.RandomCrop(384),
                                  transforms.ToTensor(),
                                  normalize,
                              ]))

    # 测试集图片变换
    test_data = TestDataset(test_data_list,
                            transform=transforms.Compose([
                                transforms.Resize((400, 400)),
                                transforms.CenterCrop(384),
                                transforms.ToTensor(),
                                normalize,
                            ]))

    # 生成图片迭代器
    train_loader = DataLoader(train_data, batch_size=config.train.batch_size, shuffle=True, pin_memory=True, num_workers=config.workers)
    test_loader = DataLoader(test_data, batch_size=config.test.batch_size, shuffle=False, pin_memory=False, num_workers=config.workers)

    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss().cuda()

    # 优化器，使用带amsgrad的Adam
    optimizer = optim.Adam(model.parameters(), config.train.lr, weight_decay=config.train.weight_decay, amsgrad=True)

    best_precision = 0
    lowest_loss = 1000
    best_epoch = 0
    # optionally resume from a checkpoint
    if config.load_mode_path is not None:
        checkpoint_path = config.load_mode_path
        if os.path.isfile(checkpoint_path):
            logger.info("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            start_epoch = checkpoint['epoch'] + 1
            best_epoch = checkpoint['best_epoch']
            best_precision = checkpoint['best_precision']
            lowest_loss = checkpoint['lowest_loss']
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found")
    else:
        start_epoch = 0

    # 开始训练
    for epoch in range(start_epoch, config.train.epochs):
        # train for one epoch
        adjustLR(optimizer, epoch, config)
        train(train_loader, model, criterion, optimizer, epoch, logger)
        if config.val_ratio is not None:
            # evaluate on validation set
            precision, avg_loss = validate(val_loader, model, criterion)
            # 记录最高精度与最低loss，保存最新模型与最佳模型
            if precision >= best_precision:
                best_epoch = epoch
            best_precision = max(precision, best_precision)
            lowest_loss = min(avg_loss, lowest_loss)
            logger.info('Validation Accuray: {} (Previous Best Acc{}), Loss: {} (Previous Lowest Loss: {})'
                        .format(precision, best_precision, avg_loss, lowest_loss))
            state = {
                'epoch': epoch,
                'best_epoch': best_epoch,
                'state_dict': model.state_dict(),
                'best_precision': best_precision,
                'lowest_loss': lowest_loss,
            }
            save_checkpoint(state, os.path.join(config.exp.model_path, "{}.pth".format(epoch)))
        else:  # 使用所有数据作为训练集
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
            }
            save_checkpoint(state, os.path.join(config.exp.model_path, "all_in_{}.pth".format(epoch)))

    # 统计线下最佳分数
    if config.val_ratio is not None:
        with open(config.summary_file, 'a') as acc_file:
            acc_file.write('%s* num_classes: %d best epoch: %d best acc: %.8f lowest loss: %.8f\n' %
                           (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
                            config.train.num_classes, best_epoch, best_precision, lowest_loss))

    # 读取最佳模型，预测测试集，并生成可直接提交的结果文件
    if config.val_ratio is not None:
        best_model = torch.load(os.path.join(config.exp.model_path, "{}.pth".format(best_epoch)))
    else:
        best_model = torch.load(os.path.join(config.exp.model_path, "all_in.pth".format(epoch)))
    model.load_state_dict(best_model['state_dict'])
    test(test_loader=test_loader, model=model, out_path=os.path.join(config.exp.base, "submission.csv"))

    # 释放GPU缓存
    torch.cuda.empty_cache()
    # 归档日志
    logger.destroy()


if __name__ == '__main__':
    main()
