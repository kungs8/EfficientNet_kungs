# -*- encoding: utf-8 -*-
'''
@Time    :   2020/11/20:5:27 PM
@Author  :   yanpenggong
@Version :   1.0
@Contact :   yanpenggong@163.com
@Software:   PyCharm
@Project :   解析 tfrecord格式的数据
'''
import tensorflow as tf


def _parse_image_function(example_proto):
    """
    解析输入的tf.proto 实例
    :param example_proto:
    :return:
    """
    return tf.io.parse_single_example(serialized=example_proto,
                                      features={
                                          "label": tf.io.FixedLenFeature([], tf.dtypes.int64),
                                          "image_raw": tf.io.FixedLenFeature([], tf.dtypes.string),
                                      })


def get_parsed_dataset(tfrecord_name):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_name)
    parse_dataset = raw_dataset.map(_parse_image_function)
    return parse_dataset