import datetime
import uuid
from pathlib import Path
from typing import Dict, Sequence, Union

import numpy as np
import xmltodict as xdict


def experiment_settings(
    output_filename: Union[Path, str],
    use_uuid: bool = False,
    use_datetime: bool = True,
    kwargs: Dict = {}
    ) -> Path:
    """Function to generate an XML file containing settings 
    about a ML experiment.

    Args:
        output_filename (Union[Path, str]): Path to output xml file.
        use_uuid (bool, optional): Flag to enable adding a UUID to the 
        filename. Defaults to False.
        use_datetime (bool, optional): Flag to enable adding a datetime 
        stamp to the filename. Defaults to True.
        kwargs (Dict, optional): Dictionary containing attributes to be 
        added to the XML file. Defaults to {}.

    Returns:
        pathlib.Path: Path to generated XML file.
    """

    # Make output_filename a pathlib.Path item
    output_filename = Path(output_filename)

    # Add a datetime stamp to the filename
    if use_datetime:
        curr_date = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M")
        output_filename = output_filename.parent.joinpath(
            f"{output_filename.stem}-{curr_date}{output_filename.suffix}"
            )
    
    # Add a UUID to the filename
    if use_uuid:
        uuid_str = str(uuid.uuid4())
        output_filename = output_filename.parent.joinpath(
            f"{output_filename.stem}-{uuid_str}{output_filename.suffix}"
            )
    
    jsxml_dict = {
        "training_parameters": {
            **kwargs
        }
        }

    with open(output_filename, "w") as fr:
        fr.write(xdict.unparse(jsxml_dict, pretty=True))
        fr.close()
      
    return output_filename

def get_experiment_settings(
    xml_filename: Union[str, Path],
    ) -> Dict:
    """Function to get experiment settings from an XML file 
    generated with the `experiment_settings` function.

    Args:
        xml_filename (Union[str, Path]): Path to XML file 
        containing experiment settings.

    Returns:
        Dict: Dictionary containing experiment settings.
    """
    
    with open(xml_filename, "r") as fd:
        doc = xdict.parse(fd.read())
        xml_item = doc['training_parameters']
    
    return xml_item
    
def image_generator(
    X: Sequence,
    y: Sequence,
    ids: Sequence,
    batch_size: int,
    clip_pixels: Sequence = [],
    max_values: Union[float, int] = [],
    mean_values: Union[float, int] = [],
    std_values: Union[float, int] = [],
    normalize_labels: bool = False
    ) -> Union[np.ndarray, np.ndarray]:
    """Generator function for yielding training batches of data

    Args:
        X (Sequence): Training data.
        y (Sequence): Training labels.
        ids (Sequence): IDs out which random selections will be drawn.
        batch_size (int): Batch size.
        clip_pixels (Sequence, optional): Number of pixels to clip. Can be 
        used to match output of a CNN with no padding. Defaults to [].
        max_values (Union[float, int], optional): Max values to be used for 
        max-normalization. Defaults to [].
        mean_values (Union[float, int], optional): Mean values to be used for 
        standardization. Defaults to [].
        std_values (Union[float, int], optional): Standard deviation values to 
        be used for standardization. Defaults to [].
        normalize_labels (bool, optional): Flag to enable normalization of 
        labels. Defaults to False.

    Yields:
        Iterator[Sequence[np.ndarray, np.ndarray]]: Batch of data
    """

    
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
        if max_values:
            X_train = np.divide(X_train, max_values)

            if normalize_labels:
                y_train = np.divide(y_train, max_values)
        elif mean_values and std_values:
            X_train = np.divide(X_train - mean_values, std_values)

            if normalize_labels:
                y_train = np.divide(y_train - mean_values, std_values)

        # Make sure they're numpy arrays (as opposed to lists)
        X_train = np.expand_dims(np.array(X_train), axis=-1)
        y_train = np.expand_dims(np.array(y_train), axis=-1)

        # The generator-y part: yield the next training batch after converting to float32
        yield X_train.astype(np.float32), y_train.astype(np.float32)
