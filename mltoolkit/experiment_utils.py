import datetime
import uuid
from pathlib import Path
from typing import Dict, Union

import xmltodict as xdict


def experiment_settings(
    output_filename: Union[Path, str],
    use_uuid: bool = False,
    use_datetime: bool = True,
    kwargs: Dict = {},
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

    jsxml_dict = {"training_parameters": {**kwargs}}

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
        xml_item = doc["training_parameters"]

    return xml_item
