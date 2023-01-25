from typing import List, Tuple, Union

from keras.layers import (
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    MaxPooling2D,
)
from keras.models import Model

from ._utils import conv2d_block, res_conv2d_block


class UNET:
    def __init__(
        self,
        input_dim: Union[List, Tuple],
        n_classes: int,
        filters: int = 64,
        batch_norm: bool = True,
        dropout: float = None,
        activation: str = "relu",
        output_activation: str = "sigmoid",
    ):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.filters = filters
        self.batch_norm = batch_norm
        self.dropout = dropout
        if self.dropout is None:
            self.dropout = 0
        self.activation = activation.lower()
        self.output_activation = output_activation.lower()
        try:
            assert self.activation in ["relu", "prelu"]
            self.use_prelu = self.activation == "prelu"
        except AssertionError:
            raise ValueError(
                "Wrong activation function. \
                Choose between 'relu' and 'prelu'."
            )
        try:
            assert self.activation in ["sigmoid", "softmax"]
        except AssertionError:
            raise ValueError(
                "Wrong output activation function. \
                Choose between 'sigmoid' and 'softmax'."
            )

        self.filter_multipliers = [1, 2, 4, 8, 16]
        self.output_filters = {"sigmoid": 1, "softmax": self.n_classes}

    def build_model(
        self,
    ):
        # Dictionary to store skip connection layers
        skip_conn = {}

        # input image
        input_img = Input(self.input_dim, name="img")

        # compression path
        for idx, mpl in enumerate(self.filter_multipliers):
            x = conv2d_block(
                input_img if idx == 0 else x,
                n_filters=self.filters * mpl,
                use_prelu=self.use_prelu,
            )
            skip_conn[tuple(x.shape)[1:3]] = x  # skip connection
            if mpl != 16:
                x = MaxPooling2D((2, 2))(x)
                x = Dropout(self.dropout)(x)

        # expansive path
        for idx, mpl in enumerate(self.filter_multipliers[:-1][::-1]):
            x = Conv2DTranspose(
                self.filters * mpl, (3, 3), strides=(2, 2), padding="same"
            )(x)
            x = Concatenate()([x, skip_conn[tuple(x.shape)[1:3]]])
            x = Dropout(self.dropout)(x)
            x = conv2d_block(x, n_filters=self.filters * mpl, use_prelu=self.use_prelu)

        outputs = Conv2D(
            self.output_filters[self.output_activation],
            (1, 1),
            activation=self.output_activation,
        )(x)
        model = Model(inputs=[input_img], outputs=[outputs])

        return model


class ResUNET:
    def __init__(
        self,
        input_dim: Union[List, Tuple],
        filters: int = 48,
        batch_norm: bool = True,
        dropout: float = None,
    ):
        self.input_dim = input_dim
        self.filters = filters
        self.batch_norm = batch_norm
        self.dropout = dropout
        if self.dropout is None:
            self.dropout = 0
        self.filter_multipliers = [2, 4, 8, *[10] * 3]

    def build_model(
        self,
    ):
        """
        Function to create a ResUNET model.
        """
        skip_conn = {}

        # input image
        input_img = Input(self.input_dim, name="img")

        # contracting path
        for i, mpl in enumerate(self.filter_multipliers):
            if i == 0:
                x = conv2d_block(
                    input_img if i == 0 else x,
                    n_filters=48,
                    batchnorm=self.batch_norm,
                    n_blocks=1,
                )
            else:
                x = res_conv2d_block(
                    input_img if i == 0 else x,
                    n_filters=self.filters * mpl,
                    dif_stride=True,
                )
            x = Dropout(self.dropout)(x)
            x = res_conv2d_block(
                x, n_filters=self.filters * mpl, dif_stride=False
            )  # skip connection here
            skip_conn[tuple(x.shape)[1:3]] = x

        # expansive path
        for i, mpl in enumerate(self.filter_multipliers[1:][::-1]):
            x = Conv2DTranspose(
                self.filters * mpl, (3, 3), strides=(2, 2), padding="same"
            )(x)
            x = Concatenate()([x, skip_conn[tuple(x.shape)[1:3]]])
            x = Dropout(self.dropout)(x)
            x = conv2d_block(x, n_filters=self.filters * mpl, batchnorm=self.batch_norm)

        outputs = Conv2D(1, (1, 1), activation="softmax")(x)
        model = Model(inputs=[input_img], outputs=[outputs])

        return model
