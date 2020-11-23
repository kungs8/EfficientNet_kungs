# -*- encoding: utf-8 -*-
'''
@Time    :   2020/11/22:10:21 PM
@Author  :   yanpenggong
@Version :   1.0
@Contact :   yanpenggong@163.com
@Software:   PyCharm
@Project :   
'''
import random
import tensorflow as tf

from configuration import train_dir, valid_dir, test_dir, train_tfrecord, valid_tfrecord, test_tfrecord
from prepare_data import get_images_and_labels


def shuffle_dict(original_dict):
    """
    将 dict 类型的数据打乱
    :param original_dict: 原始的字典数据
    :return:
    """
    keys = []
    shuffled_dict = {}
    for k in original_dict.keys():
        keys.append(k)
    random.shuffle(keys)
    for item in keys:
        shuffled_dict[item] = original_dict[item]
    return shuffled_dict


def _float_feature(value):
    """
    将值转换为类型兼容的tf.train.Feature -> float
    :param value:
    :return: An float_list from a bool / enum / int / uint.
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """
    将值转换为类型兼容的tf.train.Feature -> int64
    :param value:
    :return: An int64_list from a bool / enum / int / uint.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """
    将值转换为类型兼容的tf.train.Feature -> bytes
    :param value:
    :return: A bytes_list from a string / byte.
    """
    if isinstance(value, type(tf.constant(0.))):
        value = value.numpy()  # BytesList不会从EagerTensor中解包字符串
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def image_example(image_string, image_label):
    """
    创建具有可能相关特征的字典
    :param image_string:
    :param image_label:
    :return:
    """
    feature = {
        "label": _int64_feature(value=image_label),
        "image_raw": _bytes_feature(value=image_string)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def dataset_to_tfrecord(dataset_dir, tfrecord_name):
    image_paths, image_labels = get_images_and_labels(data_root_dir=dataset_dir)
    image_paths_and_labels_dict = {}
    for i in range(len(image_paths)):
        image_paths_and_labels_dict[image_paths[i]] = image_labels[i]
    # 打乱dict顺序
    image_paths_and_labels_dict = shuffle_dict(original_dict=image_paths_and_labels_dict)
    # 将images特征和labels写入到指定file的tfrecord类型文件中
    with tf.io.TFRecordWriter(path=tfrecord_name) as writer:
        for image_path, image_label in image_paths_and_labels_dict.items():
            print("Writing to tfrecord: {}".format(image_path))
            image_string = open(image_path, "rb").read()
            tf_example = image_example(image_string, image_label)
            writer.write(tf_example.SerializeToString())  # 序列化写文件


if __name__ == '__main__':
    dataset_to_tfrecord(dataset_dir=train_dir, tfrecord_name=train_tfrecord)
    dataset_to_tfrecord(dataset_dir=valid_dir, tfrecord_name=valid_tfrecord)
    dataset_to_tfrecord(dataset_dir=test_dir, tfrecord_name=test_tfrecord)