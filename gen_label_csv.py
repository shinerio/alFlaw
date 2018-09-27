import os
import pandas as pd
import os.path as osp
from config import config


label_warp = {'正常': 0,
              '不导电': 1,
              '擦花': 2,
              '横条压凹': 3,
              '桔皮': 4,
              '漏底': 5,
              '碰伤': 6,
              '起坑': 7,
              '凸粉': 8,
              '涂层开裂': 9,
              '脏点': 10,
              '其他': 11,
              }

binary_lable_warp ={'正常': 0,
                    '不导电': 1,
                    '擦花': 1,
                    '横条压凹': 1,
                    '桔皮': 1,
                    '漏底': 1,
                    '碰伤': 1,
                    '起坑': 1,
                    '凸粉': 1,
                    '涂层开裂': 1,
                    '脏点': 1,
                    '其他': 1,
                    }

# train data
img_path, label = [], []

for data_path in config.train.imageDir_list:
    for first_path in os.listdir(data_path):
        first_path = osp.join(data_path, first_path)
        if ".jpg" in first_path:
            if "凸粉" in first_path:
                img_path.append(osp.join(first_path))
                label.append('凸粉')
            if "擦花" in first_path:
                img_path.append(osp.join(first_path))
                label.append('擦花')
            if "漏底" in first_path:
                img_path.append(osp.join(first_path))
                label.append('漏底')
            if "碰凹" in first_path:
                img_path.append(osp.join(first_path))
                label.append('其他')
        elif '无瑕疵样本' in first_path:
            for img in os.listdir(first_path):
                img_path.append(osp.join(first_path, img))
                label.append('正常')
        else:
            for second_path in os.listdir(first_path):
                defect_label = second_path
                second_path = osp.join(first_path, second_path)
                if defect_label != '其他':
                    for img in os.listdir(second_path):
                        img_path.append(osp.join(second_path, img))
                        label.append(defect_label)
                else:
                    for third_path in os.listdir(second_path):
                        third_path = osp.join(second_path, third_path)
                        if osp.isdir(third_path):
                            for img in os.listdir(third_path):
                                if 'DS_Store' not in img:
                                    img_path.append(osp.join(third_path, img))
                                    label.append(defect_label)

label_file = pd.DataFrame({'img_path': img_path, 'label': label})
label_file['label'] = label_file['label'].map(binary_lable_warp)

label_file.to_csv(config.train.label, index=False)

# test data
test_data_path = config.test.imageDir_list
all_test_img = []
test_img_path = []
for data_path in test_data_path:
    all_test_img += os.listdir(data_path)
    for img in all_test_img:
        if osp.splitext(img)[1] == '.jpg':
            test_img_path.append(osp.join(data_path, img))


test_file = pd.DataFrame({'img_path': test_img_path})
test_file.to_csv(config.test.imageList, index=False)
