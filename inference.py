from collections import OrderedDict
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
from metric.AverageMeter import AverageMeter, accuracy


def test(test_loader, model, out_path):
    csv_map = OrderedDict({'filename': [], 'probability': []})
    # switch to evaluate mode
    model.eval()
    for i, (images, filepath) in enumerate(tqdm(test_loader)):
        # batch_size, c, h, w = images.size()
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
    submission.to_csv(out_path, header=None, index=False)
    print("save result to {}".format(out_path))


# 训练函数
def train(train_loader, model, criterion, optimizer, epoch, logger):
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
        message = ('Epoch: [{0}][{1}/{2}]\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Accuray {acc.val:.3f} ({acc.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, acc=acc))
        logger.info(message)


# 验证函数
def validate(val_loader, model, criterion):
    losses = AverageMeter()
    acc = AverageMeter()
    # switch to evaluate mode
    model.eval()
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
    return acc.avg, losses.avg
