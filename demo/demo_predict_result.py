# -*- coding: utf-8 -*-
import time
import os
import pandas as pd
from model import model_v4
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from config import config
from data.TestDataset import TestDataset
from inference import test


def main(model_path, out_path):
    # 设定GPU ID
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    # 创建inception_v4模型
    model = model_v4.v4(num_classes=config.train.num_classes)
    model = torch.nn.DataParallel(model).cuda()

    # 读取测试图片列表
    test_data_list = pd.read_csv(config.test.imageList)
    # 图片归一化，由于采用ImageNet预训练网络，因此这里直接采用ImageNet网络的参数
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 测试集图片变换
    test_data = TestDataset(test_data_list,
                            transform=transforms.Compose([
                                transforms.Resize((400, 400)),
                                transforms.CenterCrop(384),
                                transforms.ToTensor(),
                                normalize,
                            ]))
    test_loader = DataLoader(test_data, batch_size=config.test.batch_size, shuffle=False, pin_memory=False,
                             num_workers=config.workers)

    model_state = torch.load(model_path)
    model.load_state_dict(model_state['state_dict'])
    test(test_loader=test_loader, model=model, out_path=out_path)

    # 释放GPU缓存
    torch.cuda.empty_cache()


if __name__ == '__main__':
    model_path = "/home/messor/data_center/alFlaw/rui.zhang/archive/12_classes_97_665/models/41.pth"
    out_path = "/home/messor/data_center/alFlaw/rui.zhang/result/{}_submission.csv"\
        .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    main(model_path, out_path)
