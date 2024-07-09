import base64
import os
from typing import List, Optional

import cv2
import numpy as np

VALID_EXTENSIONS = ["jpg", "jpeg", "png"]


def get_latest_image(dirpath: str, valid_extensions: Optional[List[str]] = None) -> str:
    """Get the latest image file in the given directory.

    Args:
        dirpath (str): The directory to search in
        valid_extensions (List[str], optional): file extensions to filter for.
            Defaults to ['jpg', 'jpeg', 'png'].

    Raises:
        ValueError: Raised if no valid images are found

    Returns:
        str: path to the latest image file
    """

    if valid_extensions is None:
        valid_extensions = VALID_EXTENSIONS

    # Get filepaths of all files and dirs in the given dir
    valid_files = [os.path.join(dirpath, filename) for filename in os.listdir(dirpath)]
    # Filter out directories, no-extension, and wrong extension files
    valid_files = [
        f
        for f in valid_files
        if "." in f and f.rsplit(".", 1)[-1] in valid_extensions and os.path.isfile(f)
    ]

    if not valid_files:
        raise ValueError(f"No valid images in {dirpath}")

    return max(valid_files, key=os.path.getmtime)


def aspect_ratio(contour: cv2.typing.MatLike) -> float:
    """
    Calculates the aspect ratio of a contour.

    Args:
        contour (cv2.typing.MatLike): The contour whose aspect ratio is to be calculated.

    Returns:
        float: The aspect ratio of the contour.
    """

    x, y, w, h = cv2.boundingRect(contour)
    return float(w) / float(h)


def image_to_base64(img: cv2.typing.MatLike, ext: str = ".jpg") -> str:
    """Converts the image to a base64-encoded string"""
    _, buffer = cv2.imencode(ext, img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")
    return img_base64


def rescale_image(image: cv2.typing.MatLike, size: int = 100) -> cv2.typing.MatLike:
    """
    Rescales an image to a given size.

    Args:
        image (cv2.typing.MatLike): The image to be rescaled.
        size (int, optional): The size to rescale the image to. Defaults to 100.

    Returns:
        cv2.typing.MatLike: The rescaled image.
    """

    try:
        height, width = image.shape
    except ValueError:
        height, width, _ = image.shape
    new_width = int((size / 100) * width)
    new_height = int((size / 100) * height)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)


def calc_rects_pos_and_size(contours: List[cv2.typing.MatLike]) -> cv2.typing.Rect:
    """
    Calculates the position and size of a rectangle that bounds a list of contours.

    Args:
        contours (List[cv2.typing.MatLike]): The list of contours.

    Returns:
        cv2.typing.Rect: The position and size of the bounding rectangle.
    """

    # Get the bounding rectangles for the similar contours
    rects = [cv2.boundingRect(contour) for contour in contours]

    # Calculate the grid position and size from the bounding rectangles
    rects_x = min(x for (x, y, w, h) in rects)
    rects_y = min(y for (x, y, w, h) in rects)
    width = max(x + w for (x, y, w, h) in rects) - rects_x
    height = max(y + h for (x, y, w, h) in rects) - rects_y

    return rects_x, rects_y, width, height


def pad_image(image: cv2.typing.MatLike, padding: int = 30) -> cv2.typing.MatLike:
    """
    Adds padding to an image.

    Args:
        image (cv2.typing.MatLike): The image to be padded.
        padding (int, optional): The amount of padding to add. Defaults to 30.

    Returns:
        cv2.typing.MatLike: The padded image.
    """

    # Calculate the average color of the image
    avg_color_per_row = np.average(image, axis=0)  # type: ignore
    avg_color = np.average(avg_color_per_row, axis=0)
    avg_color = np.uint8(avg_color)

    # Add padding to the letter image
    padded_img = cv2.copyMakeBorder(
        image,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=avg_color.tolist(),
    )  # type: ignore

    return padded_img
