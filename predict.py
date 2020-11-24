# -*- encoding: utf-8 -*-
'''
@Time    :   2020/11/23:5:07 PM
@Author  :   yanpenggong
@Version :   1.0
@Contact :   yanpenggong@163.com
@Software:   PyCharm
@Project :   对单张图像进行预测
'''
import tensorflow as tf

from configuration import save_model_dir, test_image_dir
from prepare_data import load_and_preprocess_image
from training import get_model


def get_single_picture_prediction(model, image_dir):
    """
    获取单张图像预测
    :param model:
    :param test_image_dir:
    :return:
    """
    image_tensor = load_and_preprocess_image(image_raw=tf.io.read_file(filename=image_dir, name="test_image"), data_augmentation=False)
    image = tf.expand_dims(input=image_tensor, axis=0)
    prediction = model(image, training=False)
    pred_class = tf.math.argmax(prediction, axis=-1)
    return pred_class


if __name__ == '__main__':
    # GPUs setting
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(device=gpu, enable=True)

    # 加载模型
    model = get_model()
    # model.load_weights(filepath=save_model_dir + "model")

    pred_class = get_single_picture_prediction(model, test_image_dir)
    print('pred_class:', pred_class)