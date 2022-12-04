from collections import Iterable
from typing import Sequence, Union

import numpy as np


def image_generator(
    X: Sequence,
    y: Sequence,
    ids: Sequence,
    batch_size: int,
    clip_pixels: Sequence = None,
    min_values: Union[float, int] = None,
    max_values: Union[float, int] = None,
    mean_values: Union[float, int] = None,
    std_values: Union[float, int] = None,
    normalize_labels: bool = False,
    random_rotations: bool = False,
    random_flips: bool = False,
    ) -> Union[np.ndarray, np.ndarray]:
    """Generator function for yielding training batches of data

    Args:
        X (Sequence): Training data.
        y (Sequence): Training labels.
        ids (Sequence): IDs out which random selections will be drawn.
        batch_size (int): Batch size.
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

    Yields:
        Iterator[Sequence[np.ndarray, np.ndarray]]: Batch of data
    """
    if isinstance(min_values, Iterable):
        min_values = np.array(min_values)
    if isinstance(max_values, Iterable):
        max_values = np.array(max_values)
    if isinstance(mean_values, Iterable):
        mean_values = np.array(mean_values)
    if isinstance(std_values, Iterable):
        std_values = np.array(std_values)
    
    while True:
        # Get the samples you'll use in this batch
        batch_samples = np.random.choice(ids, size=batch_size)

        # Initialise X_train and y_train arrays for this batch
        X_train = X[batch_samples]
        y_train = y[batch_samples]

        if clip_pixels:
            clip_x, clip_y = clip_pixels
            y_train = [
                img[
                    clip_y: -clip_y, clip_x: -clip_x
                    ] for img in y_train
                    ]

        # Normalize or standardize data
        if max_values is not None:
            if min_values is not None:
                X_train = np.divide(
                    X_train - min_values,
                    max_values - min_values
                    )
                if normalize_labels:
                    y_train = np.divide(
                        y_train - min_values,
                        max_values - min_values
                        )
            else:
                X_train = np.divide(X_train, max_values)
                if normalize_labels:
                    y_train = np.divide(y_train, max_values)

        elif mean_values is not None and std_values is not None:
            X_train = np.divide(X_train - mean_values, std_values)

            if normalize_labels:
                y_train = np.divide(y_train - mean_values, std_values)
        
        # Random rotations
        if random_rotations:
            n_rotations = np.random.randint(0, 4)
            if n_rotations:
                X_train = np.rot90(X_train, k=n_rotations, axes=(1, 2))
                y_train = np.rot90(y_train, k=n_rotations, axes=(1, 2))
        
        # Random flips
        if random_flips:
            flip = np.random.randint(0, 3)
            if flip:
                X_train = np.flip(X_train, axis=flip)
                y_train = np.flip(y_train, axis=flip)

        # Make sure they're numpy arrays (as opposed to lists)
        # X_train = np.expand_dims(np.array(X_train), axis=-1)
        # y_train = np.expand_dims(np.array(y_train), axis=-1)

        # yield the next training batch after converting to float32
        yield X_train.astype(np.float32), y_train.astype(np.float32)