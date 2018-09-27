from easydict import EasyDict as edict
import os
import time


config = edict()
config.person = "rui.zhang"
config.base_path = os.path.join('/home/messor/data_center/alFlaw', config.person)
if not os.path.exists(config.base_path):
    os.mkdir(config.base_path)

config.train = edict()
config.train.imageDir_list = ['/home/messor/data_center/alFlaw/guangdong_round1_train1_20180903',
                              '/home/messor/data_center/alFlaw/guangdong_round1_train2_20180916']
config.train.num_classes = 12  # 2,12
if config.train.num_classes == 2:
    config.train.label = os.path.join(config.base_path, 'binary_label.csv')
elif config.train.num_classes == 12:
    config.train.label = os.path.join(config.base_path, 'label.csv')

config.test = edict()
config.test.imageDir_list = ['/home/messor/data_center/alFlaw/guangdong_round1_test_a_20180916']
config.test.imageList = os.path.join(config.base_path, 'test.csv')

now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
config.exp = edict()
config.exp.base = os.path.join(config.base_path, now)
if not os.path.exists(config.exp.base):
    os.mkdir(config.exp.base)
config.exp.log_path = os.path.join(config.exp.base, "log.txt")
config.exp.model_path = os.path.join(config.exp.base, "models")
if not os.path.exists(config.exp.model_path):
    os.mkdir(config.exp.model_path)

config.summary_file = os.path.join(config.base_path, 'summary.txt')
config.load_mode_path = None


# 进程数量，最好不要超过电脑最大进程数，尽量能被batch size整除。windows下报错可以改为workers=0
config.workers = 12
config.val_ratio = 0.12

config.train.batch_size = 24
# epoch数量，分stage进行，跑完一个stage后降低学习率进入下一个stage
config.train.stage_epochs = [2, 3, 4]
config.train.epochs = 6
# 初始学习率
config.train.lr = 4 * 1e-6 * config.train.batch_size
# 学习率衰减系数 (new_lr = lr / lr_decay)
config.train.lr_decay = 5
# 正则化系数
config.train.weight_decay = 1e-4

config.test.batch_size = 60
