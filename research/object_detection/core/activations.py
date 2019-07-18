import tensorflow as tf


def hsigmoid(x):
    """
    hard sigmoid. Ref: https://arxiv.org/pdf/1905.02244.pdf
    :param x: input tensor
    :return: tensor applied hard sigmoid activation
    """
    return tf.nn.relu6(x + 3) / 6


def hswish(x):
    """
    Ref: https://arxiv.org/pdf/1905.02244.pdf
    :param x: input tensor
    :return: tensor applied h-swish activation
    """
    return x * hsigmoid(x)
