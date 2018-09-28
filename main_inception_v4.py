# -*- coding: utf-8 -*-
'''
Created on Thu Sep 20 16:16:39 2018
 
@ author: herbert-chen
'''
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import model_v4
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import time
from config import config
from utils.logger import Logger
import pprint


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

    # 默认使用PIL读图
    def default_loader(path):
        # return Image.open(path)
        return Image.open(path).convert('RGB')

    # 训练集图片读取
    class TrainDataset(Dataset):
        def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
            imgs = []
            for index, row in label_list.iterrows():
                imgs.append((row['img_path'], row['label']))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader

        def __getitem__(self, index):
            filename, label = self.imgs[index]
            img = self.loader(filename)
            if self.transform is not None:
                img = self.transform(img)
            return img, label

        def __len__(self):
            return len(self.imgs)

    # 验证集图片读取
    class ValDataset(Dataset):
        def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
            imgs = []
            for index, row in label_list.iterrows():
                imgs.append((row['img_path'], row['label']))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader

        def __getitem__(self, index):
            filename, label = self.imgs[index]
            img = self.loader(filename)
            if self.transform is not None:
                img = self.transform(img)
            return img, label

        def __len__(self):
            return len(self.imgs)

    # 测试集图片读取
    class TestDataset(Dataset):
        def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
            imgs = []
            for index, row in label_list.iterrows():
                imgs.append((row['img_path']))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader

        def __getitem__(self, index):
            filename = self.imgs[index]
            img = self.loader(filename)
            if self.transform is not None:
                img = self.transform(img)
            return img, filename

        def __len__(self):
            return len(self.imgs)

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

    # 训练函数
    def train(train_loader, model, criterion, optimizer, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        # switch to train mode
        model.train()

        end = time.time()
        # 从训练集迭代器中获取训练数据
        for i, (images, target) in enumerate(train_loader):
            # 评估图片读取耗时
            data_time.update(time.time() - end)
            # 将图片和标签转化为tensor
            image_var = torch.tensor(images).cuda(async=True)
            label = torch.tensor(target).cuda(async=True)

            # 将图片输入网络，前传，生成预测值
            y_pred = model(image_var)
            # 计算loss
            loss = criterion(y_pred, label)
            losses.update(loss.item(), images.size(0))

            # 计算top1正确率
            prec, PRED_COUNT = accuracy(y_pred.data, target, topk=(1, 1))
            acc.update(prec, PRED_COUNT)

            # 对梯度进行反向传播，使用随机梯度下降更新网络权重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 评估训练耗时
            batch_time.update(time.time() - end)
            end = time.time()

            # 打印耗时与结果
            if i % print_freq == 0:
                message = ('Epoch: [{0}][{1}/{2}]\t'
                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                           'Accuray {acc.val:.3f} ({acc.avg:.3f})'.format(
                            epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, acc=acc))
                logger.info(message)

    # 验证函数
    def validate(val_loader, model, criterion):
        batch_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (images, labels) in enumerate(val_loader):
            image_var = torch.tensor(images).cuda(async=True)
            target = torch.tensor(labels).cuda(async=True)

            # 图片前传。验证和测试时不需要更新网络权重，所以使用torch.no_grad()，表示不计算梯度
            with torch.no_grad():
                y_pred = model(image_var)
                loss = criterion(y_pred, target)

            # measure accuracy and record loss
            prec, PRED_COUNT = accuracy(y_pred.data, labels, topk=(1, 1))
            losses.update(loss.item(), images.size(0))
            acc.update(prec, PRED_COUNT)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                message = ('TrainVal: [{0}/{1}]\t'
                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                           'Accuray {acc.val:.3f} ({acc.avg:.3f})'.format(
                           i, len(val_loader), batch_time=batch_time, loss=losses, acc=acc))
                logger.info(message)
        logger.info(' * Accuray {acc.avg:.3f}'.format(acc=acc), '(Previous Best Acc: %.3f)' % best_precision,
                    ' * Loss {loss.avg:.3f}'.format(loss=losses), 'Previous Lowest Loss: %.3f)' % lowest_loss)
        return acc.avg, losses.avg

    # 测试函数
    def test(test_loader, model):
        csv_map = OrderedDict({'filename': [], 'probability': []})
        # switch to evaluate mode
        model.eval()
        for i, (images, filepath) in enumerate(tqdm(test_loader)):
            # bs, ncrops, c, h, w = images.size()
            filepath = [os.path.basename(i) for i in filepath]
            image_var = torch.tensor(images, requires_grad=False)  # for pytorch 0.4

            with torch.no_grad():
                y_pred = model(image_var)
                # 使用softmax函数将图片预测结果转换成类别概率
                smax = nn.Softmax(1)
                smax_out = smax(y_pred)
                
            # 保存图片名称与预测概率
            csv_map['filename'].extend(filepath)
            for output in smax_out:
                prob = ';'.join([str(i) for i in output.data.tolist()])
                csv_map['probability'].append(prob)

        result = pd.DataFrame(csv_map)
        result['probability'] = result['probability'].map(lambda x: [float(i) for i in x.split(';')])

        # 转换成提交样例中的格式
        sub_filename, sub_label = [], []
        for index, row in result.iterrows():
            sub_filename.append(row['filename'])
            pred_label = np.argmax(row['probability'])
            if pred_label == 0:
                sub_label.append('norm')
            else:
                sub_label.append('defect%d' % pred_label)

        # 生成结果文件，保存在result文件夹中，可用于直接提交
        submission = pd.DataFrame({'filename': sub_filename, 'label': sub_label})
        submission.to_csv(os.path.join(config.exp.base, 'submission.csv'), header=None, index=False)
        print("test over")
        return

    def adjustLR(optimizer, epoch):
        lr = config.train.lr
        for stage in config.train.stage_epochs:
            if epoch + 1 >= stage:
                lr /= config.train.lr_decay
        logger.info("adjust lr to {}".format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # 保存最新模型以及最优模型
    def save_checkpoint(state, filename):
        logger.info("save model to {}".format(filename))
        torch.save(state, filename)

    # 用于计算精度和时间的变化
    class AverageMeter(object):
        """Computes and stores the average and current value"""
        def __init__(self):
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

    # 计算top K准确率
    def accuracy(y_pred, y_actual, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        final_acc = 0
        maxk = max(topk)
        # for prob_threshold in np.arange(0, 1, 0.01):
        PRED_COUNT = y_actual.size(0)
        PRED_CORRECT_COUNT = 0
        prob, pred = y_pred.topk(maxk, 1, True, True)
        # prob = np.where(prob > prob_threshold, prob, 0)
        for j in range(pred.size(0)):
            if int(y_actual[j]) == int(pred[j]):
                PRED_CORRECT_COUNT += 1
        if PRED_COUNT == 0:
            final_acc = 0
        else:
            final_acc = PRED_CORRECT_COUNT / PRED_COUNT
        return final_acc * 100, PRED_COUNT

    # 设定GPU ID
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    best_precision = 0
    lowest_loss = 100

    # 设定打印频率，即多少step打印一次，用于观察loss和acc的实时变化
    # 打印结果中，括号前面为实时loss和acc，括号内部为epoch内平均loss和acc
    print_freq = 1
    # 是否只验证，不训练
    evaluate = False
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
        val_data = ValDataset(val_data_list,
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

    best_epoch = 0
    # optionally resume from a checkpoint
    if config.load_mode_path is not None:
        checkpoint_path = config.load_mode_path
        if os.path.isfile(checkpoint_path):
            logger.info("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            start_epoch = checkpoint['epoch'] + 1
            best_precision = checkpoint['best_precision']
            lowest_loss = checkpoint['lowest_loss']
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found")
    else:
        start_epoch = 0
    if evaluate:
        validate(val_loader, model, criterion)
    else:
        # 开始训练
        for epoch in range(start_epoch, config.train.epochs):

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch)
            adjustLR(optimizer, epoch)
            if config.val_ratio is not None:
                # evaluate on validation set
                precision, avg_loss = validate(val_loader, model, criterion)

                # 在日志文件中记录每个epoch的精度和loss
                logger.info('Epoch: %2d, Precision: %.8f, Loss: %.8f\n' % (epoch, precision, avg_loss))

                # 记录最高精度与最低loss，保存最新模型与最佳模型
                if precision >= best_precision:
                    best_epoch = epoch
                best_precision = max(precision, best_precision)
                lowest_loss = min(avg_loss, lowest_loss)
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_precision': best_precision,
                    'lowest_loss': lowest_loss,
                }
                save_checkpoint(state, os.path.join(config.exp.model_path, "{}.pth".format(epoch)))
            elif epoch-1 == config.train.epochs:  # 使用所有数据作为训练集
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                }
                save_checkpoint(state, os.path.join(config.exp.model_path, "all_in.pth"))
        # 记录线下最佳分数
        if config.val_ratio is not None:
            with open(config.summary_file, 'a') as acc_file:
                acc_file.write('%s* num_classes: %d best epoch: %d best acc: %.8f lowest loss: %.8f\n' %
                               (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
                                config.train.num_classes, best_epoch, best_precision, lowest_loss))

    # 读取最佳模型，预测测试集，并生成可直接提交的结果文件
    if config.val_ratio is not None:
        best_model = torch.load(os.path.join(config.exp.model_path, "{}.pth".format(best_epoch)))
    else:
        best_model = torch.load(os.path.join(config.exp.model_path, "all_in.pth"))
    model.load_state_dict(best_model['state_dict'])
    test(test_loader=test_loader, model=model)

    # 释放GPU缓存
    torch.cuda.empty_cache()
    # 归档日志
    logger.destroy()


if __name__ == '__main__':
    main()
