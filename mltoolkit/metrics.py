import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

    import numpy as np
    import tensorflow as tf
    import tensorflow.keras.backend as K

    np.random.seed(42)
    tf.random.set_seed(42)


def jaccard_distance(y_true, y_pred, smooth=100):
    """Calculates mean of Jaccard distance as a loss function"""
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))

    return 1 - numerator / denominator


def weighted_binary_cross_entropy(weights: dict, from_logits: bool = False):
    """
    Return a function for calculating weighted binary cross entropy
    It should be used for multi-hot encoded labels

    # Example
    y_true = tf.convert_to_tensor([1, 0, 0, 0, 0, 0], dtype=tf.int64)
    y_pred = tf.convert_to_tensor([0.6, 0.1, 0.1, 0.9, 0.1, 0.], dtype=tf.float32)
    weights = {
        0: 1.,
        1: 2.
    }
    # with weights
    loss_fn = get_loss_for_multilabels(weights=weights, from_logits=False)
    loss = loss_fn(y_true, y_pred)
    print(loss)
    # tf.Tensor(0.6067193, shape=(), dtype=float32)

    # without weights
    loss_fn = get_loss_for_multilabels()
    loss = loss_fn(y_true, y_pred)
    print(loss)
    # tf.Tensor(0.52158177, shape=(), dtype=float32)

    # Another example
    y_true = tf.convert_to_tensor([[0., 1.], [0., 0.]], dtype=tf.float32)
    y_pred = tf.convert_to_tensor([[0.6, 0.4], [0.4, 0.6]], dtype=tf.float32)
    weights = {
        0: 1.,
        1: 2.
    }
    # with weights
    loss_fn = get_loss_for_multilabels(weights=weights, from_logits=False)
    loss = loss_fn(y_true, y_pred)
    print(loss)
    # tf.Tensor(1.0439969, shape=(), dtype=float32)

    # without weights
    loss_fn = get_loss_for_multilabels()
    loss = loss_fn(y_true, y_pred)
    print(loss)
    # tf.Tensor(0.81492424, shape=(), dtype=float32)

    @param weights A dict setting weights for 0 and 1 label. e.g.
        {
            0: 1.
            1: 8.
        }
        For this case, we want to emphasise those true (1) label,
        because we have many false (0) label. e.g.
            [
                [0 1 0 0 0 0 0 0 0 1]
                [0 0 0 0 1 0 0 0 0 0]
                [0 0 0 0 1 0 0 0 0 0]
            ]



    @param from_logits If False, we apply sigmoid to each logit
    @return A function to calculate (weighted) binary cross entropy
    """
    assert 0 in weights
    assert 1 in weights

    def weighted_cross_entropy_fn(y_true, y_pred):
        tf_y_true = tf.cast(y_true, dtype=y_pred.dtype)
        tf_y_pred = tf.cast(y_pred, dtype=y_pred.dtype)

        weights_v = tf.where(tf.equal(tf_y_true, 1), weights[1], weights[0])
        ce = K.binary_crossentropy(tf_y_true, tf_y_pred, from_logits=from_logits)
        weights_v = tf.cast(weights_v, dtype=y_pred.dtype)
        loss = K.mean(tf.multiply(ce, weights_v))
        return loss

    return weighted_cross_entropy_fn
