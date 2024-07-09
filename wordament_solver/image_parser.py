import math
from typing import List, Tuple

import cv2
import numpy as np
import pytesseract
from scipy import stats

from wordament_solver.image_viewer import (
    display_word_image,
    draw_word_arrows,
    generate_grid_image,
    generate_word_image,
)

from .models import Cell, Puzzle
from .utils import (
    aspect_ratio,
    calc_rects_pos_and_size,
    get_latest_image,
    pad_image,
    rescale_image,
)
from .words_finder import find_words, load_word_list


def get_score(score_img: cv2.typing.MatLike) -> int:
    """
    Extracts the score from an image.

    Args:
        score_img (cv2.typing.MatLike): The image of the score.

    Returns:
        int: The score extracted from the image.
    """

    padded_score = pad_image(score_img)

    # cv2.imshow('Score Image', cell)
    # cv2.waitKey(0)

    # Rescale the image
    rescaled_image = cv2.resize(
        padded_score, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC
    )

    # Apply thresholding to the score subcell image
    _, score_thresh = cv2.threshold(
        rescaled_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Extract the score text from the thresholded image using pytesseract
    score_config = r"--psm 6 -c tessedit_char_whitelist=1234567890"
    score_text = pytesseract.image_to_string(score_thresh, config=score_config)
    score = int(score_text.strip()) if score_text.strip() else 1

    return score


def get_letter(letter_img: cv2.typing.MatLike) -> str:
    """
    Extracts the letter from an image.

    Args:
        letter_img (cv2.typing.MatLike): The image of the letter.

    Returns:
        str: The letter extracted from the image.
    """

    padded_letter = pad_image(letter_img)

    # Apply thresholding to the cell image
    _, letter_bin = cv2.threshold(
        padded_letter, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # cv2.imshow('Letter Image', letter_bin)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Extract the letter from the image
    letter_config = r"--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ/-"
    letter_text = pytesseract.image_to_string(letter_bin, config=letter_config)
    letter = letter_text.strip().upper()

    return letter


def get_cell_letter_and_score_images(
    cell: cv2.typing.MatLike,
) -> Tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
    """
    Extracts the images of the letter and score from a cell image.

    Args:
        cell (cv2.typing.MatLike): The image of the cell.

    Returns:
        Tuple[cv2.typing.MatLike, cv2.typing.MatLike]: A tuple containing the images of the letter and score.
    """

    cell = rescale_image(cell, 180)

    # Color the grayscale image
    cell_color = cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR)

    # Cell Dimensions
    height, width = cell.shape

    # Apply adaptive thresholding
    letter_bin = cv2.adaptiveThreshold(
        cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Dilate the image
    kernel = np.ones((3, 3), np.uint8)
    letter_bin = cv2.dilate(letter_bin, kernel, iterations=1)

    # Find contours
    letter_contours, hierarchy = cv2.findContours(
        letter_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    # Initialize empty lists to hold the letter and score contours
    letter_contours_list = []
    score_contours_list = []

    # Draw rectangles around the letter and score
    if letter_contours:
        for i, contour in enumerate(letter_contours):
            # Filter contours based on size
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                # Check if this contour is the outermost one (i.e., it has no parent)
                if hierarchy[0][i][3] == -1:
                    # Ignore the largest contour that covers the cell
                    if w < width * 0.9 and h < height * 0.9:
                        # If the contour is in the upper left part of the cell and its height is less than half of the cell height, it's likely to be a score
                        if y < height * 0.3 and h < height * 0.3 and x < width * 0.5:
                            score_contours_list.append(contour)
                            # Green rectangle for score
                            # cv2.rectangle(cell_color, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        # If the contour is positioned in the midsection of the image
                        elif y < height * 0.9 and y > height * 0.2:
                            letter_contours_list.append(contour)
                            # Red rectangle for letter
                            # cv2.rectangle(cell_color, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Get the position and size of the letters
    ltr_x, ltr_y, ltr_w, ltr_h = calc_rects_pos_and_size(letter_contours_list)

    # Red rectangle for letter
    # cv2.rectangle(cell_color, (ltr_x, ltr_y),
    #               (ltr_x+ltr_w, ltr_y+ltr_h), (0, 0, 255), 2)

    cropped_letter = cell[ltr_y : ltr_y + ltr_h, ltr_x : ltr_x + ltr_w]

    # Get the position and size of the scores
    scr_x, scr_y, scr_w, scr_h = calc_rects_pos_and_size(score_contours_list)

    # Green rectangle for score
    # cv2.rectangle(cell_color, (scr_x, scr_y),
    #               (scr_x+scr_w, scr_y+scr_h), (0, 255, 0), 2)

    cropped_score = cell[scr_y : scr_y + scr_h, scr_x : scr_x + scr_w]

    # cv2.imshow('Cell Image', cell_color)
    # cv2.imshow('Cropped Letter', cropped_letter)
    # cv2.imshow('Cropped Score', cropped_score)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return cropped_letter, cropped_score


def get_grid_data(grid: cv2.typing.MatLike) -> List[List[Cell]]:
    """
    Extracts the data from a grid of cells.

    Args:
        grid (cv2.typing.MatLike): The image of the grid.

    Returns:
        List[List[Cell]]: A 2D list of Cell objects representing the grid.
    """

    # Split the cropped image into individual cells
    grid_size = 4  # Fixed grid size of 4x4
    grid_height, grid_width = grid.shape
    cell_size = grid_height / grid_size, grid_width / grid_size
    cells = [
        grid[int(y) : int(y + cell_size[1]), int(x) : int(x + cell_size[0])]
        for y in range(0, math.ceil(grid_height), math.ceil(cell_size[1]))
        for x in range(0, math.ceil(grid_width), math.ceil(cell_size[0]))
    ]
    # print(f'Number of cells before validation: {len(cells)}')

    # Validate cells
    valid_cells = []
    for cell in cells:
        cell_height, cell_width = cell.shape
        cell_aspect_ratio = cell_width / cell_height
        # print(f'Height: {cell_height}, Width: {cell_width}')
        if 0.8 < cell_aspect_ratio < 1.2:
            valid_cells.append(cell)
    # print(f'Number of cells after validation: {len(valid_cells)}')

    cells_list = []

    for cell in valid_cells:
        letter_img, score_img = get_cell_letter_and_score_images(cell)
        # Get the score for the cell
        score = get_score(score_img)
        # print(f'Score: {score} Cell: {cell_id}')

        # Get the letter for the cell
        letter = get_letter(letter_img)

        # FIXME: Hardcoded I in case it does not detect letter
        # based on testing its most likely I but not always
        if letter == "":
            letter = "I"
        # print(f'Letter: {letter}')

        # print(f'Letter: {letter}, Score: {score}, Cell: {cell_id}')

        # Create a Cell object and add it to the list
        cell = Cell(letter, score)
        cells_list.append(cell)

    # Create a grid of Cell objects
    grid_cells = [cells_list[i : i + 4] for i in range(0, len(cells_list), 4)]

    return grid_cells


def get_grid(
    file_path: str,
    cropped: bool = False,
    size_ratio_threshold: float = 20000,
    aspect_ratio_threshold: tuple = (0.8, 1.2),
) -> cv2.typing.MatLike:
    """Reads an image file, converts it to grayscale, and crops it to isolate the puzzle grid.

    Args:
        filename (str): image file path
        cropped (bool): whether the image is already cropped or not
        size_ratio_threshold (float): minimum contour area for filtering
        aspect_ratio_threshold (tuple): desired range for aspect ratio filtering

    Returns:
        cv2.typing.MatLike: cropped image of the puzzle grid

    Raises:
        ValueError: No contours found in the image.
    """
    screenshot = cv2.imread(file_path)

    screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    if not cropped:
        # Apply thresholding
        _, thresholded_image = cv2.threshold(
            screenshot_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Find contours
        contours, _ = cv2.findContours(
            thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours based on size
        filtered_contours = [
            contour
            for contour in contours
            if cv2.contourArea(contour) > size_ratio_threshold
        ]

        # Calculate the aspect ratio for each contour and filter them based on the desired range
        aspect_ratios = [aspect_ratio(contour) for contour in filtered_contours]

        valid_contours = [
            contour
            for contour, aspect in zip(filtered_contours, aspect_ratios)
            if aspect_ratio_threshold[0] < aspect < aspect_ratio_threshold[1]
        ]
        # Calculate the area of each contour
        areas = [cv2.contourArea(contour) for contour in valid_contours]

        # Check if areas is not empty
        if not areas:
            raise ValueError(
                "No contours found in the image. Please adjust the size_ratio_threshold or aspect_ratio_threshold arguments."
            )

        # Find the most common area size
        most_common_area = stats.mode(areas)
        most_common_area = most_common_area.mode

        # Keep only the contours that are close to the most common size
        # You may need to adjust the threshold depending on your specific images
        threshold = most_common_area * 0.1
        similar_contours = [
            contour
            for contour in valid_contours
            if abs(cv2.contourArea(contour) - most_common_area) < threshold
        ]

        # Get the bounding rectangles for the similar contours
        rects = [cv2.boundingRect(contour) for contour in similar_contours]

        # Calculate the grid position and size from the bounding rectangles
        grid_x = min(x for (x, y, w, h) in rects)
        grid_y = min(y for (x, y, w, h) in rects)
        grid_w = max(x + w for (x, y, w, h) in rects) - grid_x
        grid_h = max(y + h for (x, y, w, h) in rects) - grid_y

        # Crop the image
        cropped_img = screenshot_gray[
            grid_y : grid_y + grid_h, grid_x : grid_x + grid_w
        ]
    else:
        cropped_img = screenshot_gray

    return cropped_img


if __name__ == "__main__":
    latest_image = get_latest_image("data/")

    # Load the cropped image of the grid using OpenCV
    grid = get_grid(latest_image)

    # cv2.imshow('Grid', grid)

    cells = get_grid_data(grid)

    # Create a Puzzle object from the extracted data
    puzzle = Puzzle(cells)
    print(puzzle)

    # Find all valid words in the puzzle
    trie = load_word_list()
    words = find_words(puzzle, trie)

    # Print the highest-scoring words
    print("Highest-scoring words:")
    for word, score, path in words:
        print(f"{word}: {score}")

    img = generate_grid_image(puzzle)

    for i, word_data in enumerate(words):
        word_img = img.copy()  # Create a copy of the original image for each word
        word_img = draw_word_arrows(word_data, word_img)
        word_img = generate_word_image(word_data, word_img)

        display_word_image(word_data, word_img)
