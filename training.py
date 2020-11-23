# -*- encoding: utf-8 -*-
'''
@Time    :   2020/11/20:5:12 PM
@Author  :   yanpenggong
@Version :   1.0
@Contact :   yanpenggong@163.com
@Software:   PyCharm
@Project :   
'''
import math

import tensorflow as tf
from configuration import model_index, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, EPOCHS, BATCH_SIZE, save_every_n_epoch, \
    save_model_dir
from prepare_data import generate_datasets, load_and_preprocess_image
from models import efficientnet


def get_model():
    if model_index == 0:
        return efficientnet.EfficientNetB0()
    elif model_index == 1:
        return efficientnet.EfficientNetB1()
    elif model_index == 2:
        return efficientnet.EfficientNetB2()
    elif model_index == 3:
        return efficientnet.EfficientNetB3()
    elif model_index == 4:
        return efficientnet.EfficientNetB4()
    elif model_index == 5:
        return efficientnet.EfficientNetB5()
    elif model_index == 6:
        return efficientnet.EfficientNetB6()
    elif model_index == 7:
        return efficientnet.EfficientNetB7()
    else:
        raise ValueError("The model_index does not exist!")


def print_model_summary(network):
    """
    打印模型的summary
    :param network: 网络模型
    :return:
    """
    network.build(input_shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    network.summary()


def process_features(features, data_augmentation):
    image_raw = features["image_raw"].numpy()
    image_tensor_list = []
    for image in image_raw:
        image_tensor = load_and_preprocess_image(image, data_augmentation=data_augmentation)
        image_tensor_list.append(image_tensor)
    images = tf.stack(image_tensor_list, axis=0)
    labels = features["label"].numpy()
    return images, labels


def train_step(image_batch, label_batch):
    """
    训练数据
    :param image_batch: 图像特征的批次
    :param label_batch: 标签的批次
    :return:
    """
    with tf.GradientTape() as tape:
        predictions = model(image_batch, training=True)
        loss = loss_project(y_true=label_batch, y_pred=predictions)
    gradients = tape.gradient(loss, model.trainable_variables)  # 根据tape上面的上下文来计算某个或者某些tensor的梯度
    optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))  # 把计算出来的梯度更新到变量上面去。

    train_loss.update_state(values=loss)
    train_accuracy.update_state(y_true=label_batch, y_pred=predictions)


def valid_step(image_batch, label_batch):
    """
    验证数据
    :param valid_images:
    :param valid_labels:
    :return:
    """
    predictions = model(image_batch, training=False)
    v_loss = loss_project(y_true=label_batch, y_pred=predictions)

    valid_loss.update_state(values=v_loss)
    valid_accuracy.update_state(y_true=label_batch, y_pred=predictions)


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.list_physical_devices("GPU")
    print("gpus:", gpus)
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPUs devices!")

    # 获取数据
    train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()
    print("train_count:{}, valid_count:{}, test_count:{}".format(train_count, valid_count, test_count))

    # 创建模型
    model = get_model()
    # 打印模型的summary
    print_model_summary(network=model)

    # 定义损失函数和优化器
    loss_project = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.RMSprop()

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

    valid_loss = tf.keras.metrics.Mean(name="valid_loss")
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="valid_accuracy")

    # 开始训练
    for epoch in range(EPOCHS):
        step = 0
        # 训练数据
        for features in train_dataset:
            step += 1
            images, labels = process_features(features, data_augmentation=True)
            train_step(image_batch=images, label_batch=labels)
            print("Epoch: {}/{}, step{}/{}, loss:{:.5f}, accuracy:{:.5f}".format(epoch,
                                                                                 EPOCHS,
                                                                                 step,
                                                                                 math.ceil(train_count / BATCH_SIZE),
                                                                                 train_loss.result().numpy(),
                                                                                 train_accuracy.result().numpy()))
        # 验证数据
        for features in valid_dataset:
            valid_images, valid_labels = process_features(features, data_augmentation=False)
            valid_step(image_batch=valid_images, label_batch=valid_labels)
        print("Epoch: {}/{}, train loss:{:.5f}, accuracy:{:.5f}. Valid loss:{:.5f}, accuracy:{:.5f}".format(epoch,
                                                                                                            EPOCHS,
                                                                                                            train_loss.result().numpy(),
                                                                                                            train_accuracy.result().numpy(),
                                                                                                            valid_loss.result().numpy(),
                                                                                                            valid_accuracy.result().numpy()))
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()

        if epoch % save_every_n_epoch == 0:
            model.save_weights(filepath=save_model_dir + "epoch-{}".format(epoch), save_format="tf")

    # save weights
    model.save_weights(filepath=save_model_dir+"model", save_format="tf")
    print("model save over!")

    # save the whole model
    tf.saved_model.save(obj=model, export_dir=save_model_dir)
    print("Save whole model over!")

    # 转换为tensorflow lite 格式
    model._set_inputs(inputs=tf.random.normal(shape=(1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open("converted_model.tfiles", "wb").write(tflite_model)
    print("Convert tensorflow lite over!")