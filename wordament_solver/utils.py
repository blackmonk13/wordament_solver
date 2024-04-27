import os
from typing import Tuple


def get_latest_image(dirpath: str, valid_extensions: Tuple[str] = ('jpg', 'jpeg', 'png')) -> str:
    """Get the latest image file in the given directory.

    Args:
        dirpath (str): The directory to search in
        valid_extensions (Tuple[str], optional): file extensions to filter for. Defaults to ('jpg', 'jpeg', 'png').

    Raises:
        ValueError: Raised if no valid images are found

    Returns:
        str: path to the latest image file
    """
    
    # Get filepaths of all files and dirs in the given dir
    valid_files = [os.path.join(dirpath, filename)
                   for filename in os.listdir(dirpath)]
    # Filter out directories, no-extension, and wrong extension files
    valid_files = [f for f in valid_files if '.' in f and
                   f.rsplit('.', 1)[-1] in valid_extensions and os.path.isfile(f)]

    if not valid_files:
        raise ValueError("No valid images in %s" % dirpath)

    return max(valid_files, key=os.path.getmtime)
