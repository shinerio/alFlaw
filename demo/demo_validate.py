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
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from config import config
from TrainDataset import TrainDataset
from inference import validate


def main(model_path):
    # 随机种子
    seed = 666
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    # 设定GPU ID
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    # 创建inception_v4模型
    model = model_v4.v4(num_classes=config.train.num_classes)
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    # 读取训练图片列表
    all_data = pd.read_csv(config.train.label)
    # 图片归一化，由于采用ImageNet预训练网络，因此这里直接采用ImageNet网络的参数
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss().cuda()

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
        precision, avg_loss = validate(val_loader, model, criterion)
        print('Validation Accuray: {} , Loss: {} '.format(precision, avg_loss))
    # 释放GPU缓存
    torch.cuda.empty_cache()


if __name__ == '__main__':
    model_path = "/home/messor/data_center/alFlaw/rui.zhang/archive/12_classes_97_665/models/41.pth"
    main(model_path)
