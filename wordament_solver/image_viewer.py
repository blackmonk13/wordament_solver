from typing import List, Tuple

import cv2
import numpy as np

from .models import Puzzle


def generate_grid_image(puzzle: Puzzle) -> cv2.typing.MatLike:
    """takes a Puzzle object as input and generates 
    an image of the puzzle grid. It draws the grid lines, 
    and for each cell, it draws the letter and score.

    Args:
        puzzle (Puzzle): the puzzle to generate the image for

    Returns:
        cv2.typing.MatLike: the image of the puzzle grid
    """
    cell_size = 70  # Size of each cell in pixels
    img_size = cell_size * puzzle.nrows, cell_size * puzzle.ncols
    img = np.full(img_size, 255, dtype=np.uint8)  # Create a white image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    score_font_scale = 0.5
    score_font_thickness = 1

    for row in range(puzzle.nrows):
        for col in range(puzzle.ncols):
            cell = puzzle[row, col]

            # Calculate the text size and position to center it in the cell
            text_size, baseline = cv2.getTextSize(
                cell.letter, font, font_scale, font_thickness)
            text_width, text_height = text_size
            text_x = (col * cell_size + cell_size // 2) - text_width // 2
            text_y = (row * cell_size + cell_size // 2) + text_height // 2

            cv2.putText(img, cell.letter, (text_x, text_y), font,
                        font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

            # Calculate the score text size and position to place it in the top left corner
            score_text = str(cell.score)
            score_text_size, _ = cv2.getTextSize(
                score_text, font, score_font_scale, score_font_thickness)
            score_text_width, score_text_height = score_text_size
            score_x = col * cell_size + score_text_width + 2
            score_y = row * cell_size + score_text_height + 2

            cv2.putText(img, score_text, (score_x, score_y), font,
                        score_font_scale, (0, 0, 0), score_font_thickness, cv2.LINE_AA)

            # Draw grid lines around each cell
            top_left = (col * cell_size, row * cell_size)
            bottom_right = ((col + 1) * cell_size, (row + 1) * cell_size)
            cv2.rectangle(img, top_left, bottom_right, (0, 0, 0), 1)

    return img


def draw_word_arrows(word_data: Tuple[str, int, List[Tuple[int, int]]], img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """ takes a tuple containing a word, its score, 
    and the path of cells it covers in the puzzle, 
    and an image of the puzzle grid. It draws arrows 
    on the image to represent the path of the word 
    in the puzzle.

    Args:
        word_data (Tuple[str, int, List[Tuple[int, int]]]): tuple containing a word, its score, and the path of cells it covers in the puzzle
        img (cv2.typing.MatLike): image of the puzzle grid

    Returns:
        cv2.typing.MatLike: the image of the puzzle grid with arrows drawn on it
    """
    word, score, path = word_data
    arrow_color = (0, 0, 255)  # Blue color for the arrow
    arrow_thickness = 2

    for i in range(len(path) - 1):
        start_row, start_col = path[i]
        end_row, end_col = path[i + 1]

        start_point = (start_col * 70 + 35, start_row * 70 + 35)
        end_point = (end_col * 70 + 35, end_row * 70 + 35)

        cv2.arrowedLine(img, start_point, end_point,
                        arrow_color, arrow_thickness)

    return img


def generate_word_image(word_data: Tuple[str, int, List[Tuple[int, int]]], img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """takes a tuple containing a word, its score, and the path of cells 
    it covers in the puzzle, and an image of the puzzle grid. 
    It creates a larger image with the grid image as the background, 
    adds the word and its score to the larger image, and returns the image.

    Args:
        word_data (Tuple[str, int, List[Tuple[int, int]]]): tuple containing a word, its score, and the path of cells it covers in the puzzle
        img (cv2.typing.MatLike): image of the puzzle grid

    Returns:
        cv2.typing.MatLike: word image
    """
    word, score, _ = word_data
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    line_type = cv2.LINE_AA

    # Calculate the text size and position for the word and score
    word_text_size, _ = cv2.getTextSize(word, font, font_scale, font_thickness)
    score_text_size, _ = cv2.getTextSize(
        str(score), font, font_scale, font_thickness)

    # Stack the word and score vertically with more spacing between them
    text_width = max(word_text_size[0], score_text_size[0])
    # Increase the spacing between the word and score
    text_height = word_text_size[1] + score_text_size[1] + 30

    # Add spacing between the grid and both the word and score
    grid_spacing = 10
    text_height += grid_spacing
    text_width += grid_spacing

    # Create a larger image with the grid image as the background
    result_img = np.full(
        (text_height + img.shape[0], text_width + img.shape[1], 3), 255, dtype=np.uint8)

    # Convert the img variable to a color image before assigning it to the result_img
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result_img[text_height:, :img.shape[1]] = img_color

    # Add the word and score to the larger image with the added spacing
    result_img = cv2.putText(result_img, word, (10 + grid_spacing, text_height -
                             word_text_size[1] - 5), font, font_scale, (0, 0, 0), font_thickness, line_type)
    result_img = cv2.putText(result_img, f'Score: {
                             score}', (10 + grid_spacing, text_height - 5), font, font_scale, (0, 0, 0), font_thickness, line_type)

    # Draw a border line around the entire grid (fix the offset)
    border_thickness = 2
    border_color = (0, 0, 0)
    top_left = (grid_spacing, text_height)
    bottom_right = (
        grid_spacing + img.shape[1] - 1, text_height + img.shape[0] - 1)
    cv2.rectangle(result_img, top_left, bottom_right,
                  border_color, border_thickness)

    return result_img


def display_word_image(word_data: Tuple[str, int, List[Tuple[int, int]]], img: cv2.typing.MatLike) -> None:
    word, score, _ = word_data
    cv2.imshow(f"{word} - {score}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows
