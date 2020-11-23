# -*- encoding: utf-8 -*-
'''
@Time    :   2020/11/20:5:17 PM
@Author  :   yanpenggong
@Version :   1.0
@Contact :   yanpenggong@163.com
@Software:   PyCharm
@Project :   数据集的准备
'''
import pathlib

import tensorflow as tf
from kungs_demo.EfficientNet.EfficientNet_kungs.parse_tfrecord import get_parsed_dataset
from configuration import BATCH_SIZE, train_tfrecord, valid_tfrecord, test_tfrecord, CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT


def get_the_length_of_dataset(dataset):
    print("ttttttt:", dataset)
    count = 0
    for i in dataset:
        count += 1
    return count


def generate_datasets():
    train_dataset = get_parsed_dataset(tfrecord_name=train_tfrecord)
    valid_dataset = get_parsed_dataset(tfrecord_name=valid_tfrecord)
    test_dataset = get_parsed_dataset(tfrecord_name=test_tfrecord)

    train_count = get_the_length_of_dataset(train_dataset)
    valid_count = get_the_length_of_dataset(valid_dataset)
    test_count = get_the_length_of_dataset(test_dataset)

    # 以批处理形式读取数据集
    train_dataset = train_dataset.batch(batch_size=BATCH_SIZE)
    valid_dataset = valid_dataset.batch(batch_size=BATCH_SIZE)
    test_dataset = test_dataset.batch(batch_size=BATCH_SIZE)
    return train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count


def load_and_preprocess_image(image_raw, data_augmentation=False):
    """
    加载数据并进行预处理
    :param image_raw: 图像的原始数据矩阵
    :param data_augmentation: 数据是否增强
    :return:
    """
    # decode解码
    image_tensor = tf.io.decode_image(contents=image_raw, channels=CHANNELS, dtype=tf.dtypes.float32)

    if data_augmentation:
        image = tf.image.random_flip_left_right(image=image_tensor)
        image = tf.image.resize_with_crop_or_pad(image=image,
                                                 target_height=int(IMAGE_HEIGHT * 1.2),
                                                 target_width=int(IMAGE_WIDTH * 1.2))
        image = tf.image.random_crop(value=image, size=[IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS])
        image = tf.image.random_brightness(image=image, max_delta=0.5)
    else:
        image = tf.image.resize(images=image_tensor, size=[IMAGE_HEIGHT, IMAGE_WIDTH])
    return image


def get_images_and_labels(data_root_dir):
    """
    获取图像数据和标签
    :param dataset_dir: 数据的路径
    :return:
    """
    # 获取所有images 的路径(format: string)
    data_root = pathlib.Path(data_root_dir)
    all_image_path = [str(path) for path in list(data_root.glob("*/*"))]
    # 获取labels的名称
    label_names = sorted(item.name for item in data_root.glob("*/"))
    # dict{label: index}
    label_to_index = dict((label, index) for index, label in enumerate(label_names))
    # 获取所有的images的labels
    all_image_label = [label_to_index[pathlib.Path(single_image_path).parent.name] for single_image_path in all_image_path]
    return all_image_path, all_image_label