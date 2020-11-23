# -*- encoding: utf-8 -*-
'''
@Time    :   2020/11/20:5:34 PM
@Author  :   yanpenggong
@Version :   1.0
@Contact :   yanpenggong@163.com
@Software:   PyCharm
@Project :   配置文件的信息配置
'''

# 一些训练的参数
BATCH_SIZE = 16
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
CHANNELS = 3
EPOCHS = 1

# 保存模型间隔epoch数
save_every_n_epoch = 10
save_model_dir = "./saved_models/"

# 训练集，验证集和测试集的比例
TRAIN_SET_RATIO = 0.8
TEST_SET_RATIO = 0.1
# VALID_SET_RATIO = 1 - TRAIN_SET_RATIO - TEST_SET_RATIO

dataset_dir = "./dataset/"
train_dir = dataset_dir + "train"
valid_dir = dataset_dir + "valid"
test_dir = dataset_dir + "test"

train_tfrecord = dataset_dir + "train.tfrecord"
valid_tfrecord = dataset_dir + "valid.tfrecord"
test_tfrecord = dataset_dir + "test.tfrecord"


# 模型的index
model_index = 3