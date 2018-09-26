from easydict import EasyDict as edict
import os
import time


config = edict()
config.person = "rui.zhang"
config.base_path = os.path.join('/home/messor/data_center/alFlaw', config.person)
if not os.path.exists(config.base_path):
    os.mkdir(config.base_path)

config.train = edict()
config.train.imageDir_list = [# '/home/messor/data_center/alFlaw/guangdong_round1_train1_20180903',
                              '/home/messor/data_center/alFlaw/guangdong_round1_train2_20180916']
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


