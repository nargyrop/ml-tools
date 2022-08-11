from typing import List, Tuple, Union

from keras.layers import (Add, Conv2D, Conv2DTranspose, Dropout, Input,
                          MaxPooling2D)
from keras.models import Model

from dl_tools.utils import conv2d_block


class RedNetModel:
    def __init__(
        self,
        input_dim: Union[List, Tuple],
        filters: int = 64,
        batch_norm: bool = True,
        dropout: float = None,
        activation: str = 'relu'
        ) -> None:

        self.input_dim = input_dim
        self.filters = filters
        self.batch_norm = batch_norm
        self.dropout = dropout or 0
        self.activation = activation.lower()
        try:
            assert self.activation in ['relu', 'prelu']
            self.use_prelu = self.activation == 'prelu'
        except AssertionError:
            raise ValueError("Wrong activation function. Choose between 'relu' and 'prelu'.")

        self.filter_multipliers = [1, 2, 4]

    def build_model(
        self
    ):
        # Dictionary to store skip connection layers
        skip_conn = {}

        #input image
        input_img = Input(self.input_dim, name='img')

        # compression path
        for idx, mpl in enumerate(self.filter_multipliers):
            if idx > 0:
                x = MaxPooling2D((2, 2))(x)
            if idx < 2:
                x = conv2d_block(
                    input_img if idx == 0 else x,
                    n_filters=self.filters * mpl,
                    )
            else:
                x = Conv2D(
                    filters=self.filters * mpl,
                    kernel_size=(3, 3),
                    kernel_initializer="he_normal",
                    padding="same"
                )(x)
            skip_conn[tuple(x.shape)[1: 3]] = x  # skip connection
            
            if idx < 2:
                x = Dropout(self.dropout)(x)
        
        # expansive path
        for idx, mpl in enumerate(self.filter_multipliers[:-1][::-1]):
            x = Conv2DTranspose(self.filters * mpl, (3, 3), strides=(2, 2), padding="same")(x)
            x = conv2d_block(x, n_filters=self.filters * mpl)
            x = Add()([x, skip_conn[tuple(x.shape)[1: 3]]])
            x = Dropout(self.dropout)(x)

        # output layer
        outputs = Conv2D(1, (5, 5), activation='relu') (x)
        model = Model(inputs=[input_img], outputs=[outputs])

        return model
