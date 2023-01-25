from collections import Iterable
from typing import Sequence, Union

import numpy as np


def image_generator(
    x: Sequence,
    y: Sequence,
    ids: Sequence = None,
    batch_size: int = 2,
    clip_pixels: Sequence = None,
    min_values: Union[float, int] = [],
    max_values: Union[float, int] = [],
    mean_values: Union[float, int] = [],
    std_values: Union[float, int] = [],
    normalize_labels: bool = False,
    random_rotations: bool = False,
    random_flips: bool = False,
) -> Union[np.ndarray, np.ndarray]:
    """Generator function for yielding training batches of data

    Args:
        X (Sequence): Training data.
        y (Sequence): Training labels.
        ids (Sequence): IDs out which random selections will be drawn.
        Defaults to None.
        batch_size (int): Batch size. Defaults to 2.
        clip_pixels (Sequence, optional): Number of pixels to clip. Can be
        used to match output of a CNN with no padding. Defaults to [].
        min_values (Union[float, int], optional): Min values to be used for
        normalization. Defaults to [].
        max_values (Union[float, int], optional): Max values to be used for
        normalization. Defaults to [].
        mean_values (Union[float, int], optional): Mean values to be used for
        standardization. Defaults to [].
        std_values (Union[float, int], optional): Standard deviation values to
        be used for standardization. Defaults to [].
        normalize_labels (bool, optional): Flag to enable normalization of
        labels. Defaults to False.
        random_rotations (bool, optional): Flag to enable randomly rotating the samples in multiples of 90deg.
        Defaults to False.
        random_flips (bool, optional): Flag to enable randomly flipping the samples.
        Defaults to False.

    Yields:
        Iterator[Sequence[np.ndarray, np.ndarray]]: Batch of data
    """

    if ids is None:
        ids = list(range(len(x)))

    while True:
        # Get the samples you'll use in this batch
        batch_samples = np.random.choice(ids, size=batch_size)

        # Initialise X_train and y_train arrays for this batch
        x_train = x[batch_samples]
        y_train = y[batch_samples]

        if clip_pixels:
            clip_x, clip_y = clip_pixels
            y_train = [img[clip_y:-clip_y, clip_x:-clip_x] for img in y_train]

        # Normalize or standardize data
        if any(max_values):
            if any(min_values):
                x_train = np.divide(x_train - min_values, max_values - min_values)
                if normalize_labels:
                    y_train = np.divide(y_train - min_values, max_values - min_values)
            else:
                x_train = np.divide(x_train, max_values)
                if normalize_labels:
                    y_train = np.divide(y_train, max_values)

        elif any(mean_values) and any(std_values):
            x_train = np.divide(x_train - mean_values, std_values)

            if normalize_labels:
                y_train = np.divide(y_train - mean_values, std_values)

        # Random rotations
        if random_rotations:
            n_rotations = np.random.randint(0, 4)
            if n_rotations:
                x_train = np.rot90(x_train, k=n_rotations, axes=(1, 2))
                y_train = np.rot90(y_train, k=n_rotations, axes=(1, 2))

        # Random flips
        if random_flips:
            flip = np.random.randint(1, 3)
            if flip:
                x_train = np.flip(x_train, axis=flip)
                y_train = np.flip(y_train, axis=flip)

        # Make sure they're numpy arrays (as opposed to lists)
        # X_train = np.expand_dims(np.array(X_train), axis=-1)
        # y_train = np.expand_dims(np.array(y_train), axis=-1)

        # yield the next training batch after converting to float32
        yield x_train.astype(np.float32), y_train.astype(np.float32)
