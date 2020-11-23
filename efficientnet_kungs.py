# -*- encoding: utf-8 -*-
'''
@Time    :   2020/11/19:3:46 PM
@Author  :   yanpenggong
@Version :   1.0
@Contact :   yanpenggong@163.com
@Software:   PyCharm
@Project :   
'''
from keras_applications.

backend = None


# 每个blocks的参数
DEFAULT_BLOCKS_ARGS = [
    {
        "kernel_size": 3,  # 深度可分离卷积核大小 3*3
        "repeats": 1,  # 这个大结构efficientnet block重复多少次
        "filters_in": 32,  # 输入进来特征层的通道数
        "filters_out": 16,  # 本大结构输出特征层的通道数
        "expand_ratio": 1,  # 首个大结构块值为1，只有进行第一次大结构块的时候efficient block不会升维，其它次的efficient block都会升维
        "id_skip": True,
        "strides": 1,  # 输入特征层的宽、高被压缩为输入的 1/stride
        "se_ratio": 0.25
    },
    {
        "kernel_size": 3,
        "repeats": 2,
        "filters_in": 16,
        "filters_out": 24,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25
    },
    {
        "kernel_size": 5,
        "repeats": 2,
        "filters_in": 24,
        "filters_out": 40,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25
    },
    {
        "kernel_size": 3,
        "repeats": 3,
        "filters_in": 40,
        "filters_out": 80,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25
    },
    {
        "kernel_size": 5,
        "repeats": 3,
        "filters_in": 80,
        "filters_out": 112,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25
    },
    {
        "kernel_size": 5,
        "repeats": 4,
        "filters_in": 112,
        "filters_out": 192,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25
    },
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 192,
        "filters_out": 320,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25
    }
]


def swish(x):
    """
    旋转激活功能
    :param x: 输入Tensor
    :return: Swish激活(x*sigmoid(x))
    References: [激活功能的资料](https://arxiv..org/abs/1710.05941)
    """
    if backend.backend() == "tensorflow":
        try:
            return backend.tf.nn.swish(x)
        except AttributeError:
            pass
    return x * backends


def EfficientNet(width_coefficient,
                 depth_coefficient,
                 default_size,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 activation_fn=swish,
                 blocks_args=DEFAULT_BLOCKS_ARGS,
                 model_name="efficientnet",
                 include_top=True,
                 weights="imagenet",
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 classes=1000,
                 **kwargs):
    """
    使用给定的缩放系数实例化EfficientNet架构
    加载在ImageNet上预训练的权重(可选)
    注意，模型使用的数据格式约定上在Kreas配置中指定的格式，位于'～/.keras/keras.json'
    :param width_coefficient: float, 网络宽度的缩放系数
    :param depth_coefficient: float, 网络深度的缩放系数
    :param default_size: integer, 默认输入图像的大小
    :param dropout_rate: float, 最终分类器层之前的丢失率
    :param drop_connect_rate: float, 跳过连接的丢失率
    :param depth_divisor: integer, 网络宽度的单位
    :param activation: 激活功能
    :param blocks_args: 字典列表, 用于构造模块的参数
    :param model_name: string, 模型的名称
    :param include_top: 是否在网络顶层包含全连接层
    :param weights: one of: None(随机初始化), imagenet(在ImageNet上进行预训练), 加载权重文件的路径
    :param input_tensor: keras 的Tensor(即layers.input()的输出), 用作模型的图像输入
    :param input_shape: 可选的shape元组， 仅在'include_top'为False时指定. 它应该恰好具有3个输入通道
    :param pooling: 当include_top为False时用于特征提取的可选池模式.
                    - 'None', 模型的输出将是最后一个卷积层的4D Tensor输出
                    - 'avg', 全局平均池应用于最后一个卷积层的输出，因此模型的输出将为2D Tensor
                    - 'max', 应用全局最大池
    :param classes: 用于将图像分类的可选类数，仅在include_top 为True且未指定weights参数时指定
    :return: 一个Keras.Model实例
    raise:
        ValueError: 如果'weights'参数无效或输入形状无效
    """
    global backend



def EfficientNetB0(weights="imagenet",
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet()


if __name__ == '__main__':
    model = EfficientNetB0()