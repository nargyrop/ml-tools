import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Add, AveragePooling2D,
                          BatchNormalization, Conv2D, PReLU)


def conv2d_block(
        input_tensor: np.ndarray,
        n_filters: int,
        kernel_size: int = 3,
        n_blocks: int = 2,
        batchnorm: bool = True,
        use_prelu: bool = True
):
    for i in range(n_blocks):
        x = Conv2D(filters=n_filters,
                   kernel_size=(kernel_size, kernel_size),
                   kernel_initializer="he_normal",
                   padding="same")(input_tensor if i == 0 else x)
        if batchnorm:
            x = BatchNormalization()(x)
        if use_prelu:
            x = PReLU(alpha_initializer=tf.constant_initializer(0.25))(x)
        else:
            x = Activation("relu")(x)  # standard for Unet

    return x

def res_conv2d_block(
        input_tensor: np.ndarray,
        n_filters: int,
        kernel_size: int = 3,
        dif_stride: bool = False,
        n_blocks: int = 2
):
    if dif_stride:
        skip = AveragePooling2D((2, 2))(input_tensor)
    else:
        skip = input_tensor

    for i in range(n_blocks):
        x = BatchNormalization()(x if i != 0 else skip)
        x = Activation("relu")(x)  # standard for Unet
        x = Conv2D(filters=n_filters,
                   kernel_size=(kernel_size, kernel_size),
                   strides=2 if (i == 0 and dif_stride) else 1,
                   kernel_initializer="he_normal",
                   padding="same")(input_tensor if i == 0 else x)

    skip = Conv2D(n_filters, (1, 1), padding="same")(skip)
    x = Add()([skip, x])

    return x
