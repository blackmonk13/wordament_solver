import os
from typing import List, Tuple
import cv2
import pytesseract
import math
import numpy as np

from utils import get_latest_image
from words_finder import find_words, load_word_list
from models import Cell, Puzzle


def get_score(cell: cv2.typing.MatLike) -> int:
    """takes an image of a cell as input, 
    applies image processing techniques to isolate 
    the score in the cell, and uses OCR to extract 
    the score as an integer.

    Args:
        cell (cv2.typing.MatLike): cell image

    Returns:
        int: the score of the cell
    """    
    # Split the cell into a 4x4 grid
    subcell_size = cell.shape[0] // 4, cell.shape[1] // 4
    subcells = [cell[int(y):int(y+subcell_size[1]), int(x):int(x+subcell_size[0])]
                for y in range(0, cell.shape[0], subcell_size[1]) for x in range(0, cell.shape[1], subcell_size[0])]

    # Extract the score from the top left subcell
    score_subcell = subcells[0]

    # Get the dimensions of the image
    height, width = score_subcell.shape

    # Define the ROI (Region of Interest)
    # For a 65x65 image, cropping a 30x30 area from the center
    start_x = (width - (77 / 100 * width)) // 2
    start_y = (height - (46 / 100 * height)) // 2
    end_x = start_x + (95 / 100 * width)
    end_y = start_y + (92 / 100 * height)

    # Crop the image
    cropped_img = score_subcell[int(start_y):int(
        end_y), int(start_x):int(end_x)]

    # cv2.imshow('Cropped Image', cropped_img)
    # cv2.waitKey(0)

    # Rescale the image
    rescaled_image = cv2.resize(
        cropped_img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # Apply thresholding to the score subcell image
    _, score_thresh = cv2.threshold(
        rescaled_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Extract the score text from the thresholded image using pytesseract
    score_config = r'--psm 6 -c tessedit_char_whitelist=1234567890'
    score_text = pytesseract.image_to_string(score_thresh, config=score_config)
    score = int(score_text.strip()) if score_text.strip() else 1

    return score


def get_letter(cell: cv2.typing.MatLike) -> str:
    """takes an image of a cell as input, 
    applies image processing techniques to 
    isolate the letter in the cell, and uses 
    OCR to extract the letter as a string.

    Args:
        cell (cv2.typing.MatLike): cell image

    Returns:
        str: the letter of the cell
    """    
    # Get the cell dimensions
    height, width = cell.shape

    # print(f"Height: {height} x Width: {width}")

    # Define the ROI (Region of Interest)
    # Cropping an area from the center
    start_x = (4 / 100 * width)
    start_y = (23 / 100 * height)
    end_x = (width - (8 / 100 * width))
    end_y = (height - (15 / 100 * height))

    # Crop the image
    cropped_img = cell[int(start_y):int(end_y), int(start_x):int(end_x)]

    # Apply thresholding to the cell image
    _, letter_bin = cv2.threshold(
        cropped_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Extract the letter from the image
    letter_config = r'--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ/-'
    letter_text = pytesseract.image_to_string(letter_bin, config=letter_config)
    letter = letter_text.strip().upper()

    return letter


def get_grid_data(grid: cv2.typing.MatLike) -> List[List[Cell]]:
    """takes an image of the puzzle grid as input, 
    splits it into individual cells, and for each cell, 
    it calls get_score and get_letter to extract the 
    score and letter. It then creates a Cell object for 
    each cell and returns a 2D list of these objects.

    Args:
        grid (cv2.typing.MatLike): cropped image of the puzzle grid

    Returns:
        List[List[Cell]]: 2D list of Cell objects (Grid)
    """    
    # Split the cropped image into individual cells
    grid_size = 4  # Fixed grid size of 4x4
    cell_size = grid.shape[0] / grid_size, grid.shape[1] / grid_size
    cells = [grid[int(y):int(y+cell_size[1]), int(x):int(x+cell_size[0])]
             for y in range(0, math.ceil(grid.shape[0]), math.ceil(cell_size[1])) for x in range(0, math.ceil(grid.shape[1]), math.ceil(cell_size[0]))]

    cells_list = []

    for cell in cells:
        # Get the score for the cell
        score = get_score(cell)
        # print(f'Score: {score} Cell: {cell_id}')

        # Get the letter for the cell
        letter = get_letter(cell)

        # FIXME: Hardcoded I in case it does not detect letter
        # based on testing its most likely I but not always
        if letter == '':
            letter = "I"
        # print(f'Letter: {letter}')

        # print(f'Letter: {letter}, Score: {score}, Cell: {cell_id}')

        # Create a Cell object and add it to the list
        cell = Cell(letter, score)
        cells_list.append(cell)

    # Create a grid of Cell objects
    grid_cells = [cells_list[i:i+4] for i in range(0, len(cells_list), 4)]

    return grid_cells


def get_grid(file_path: str) -> cv2.typing.MatLike:
    """reads an image file, converts it to grayscale, 
    and crops it to isolate the puzzle grid.

    Args:
        filename (str): image file path

    Returns:
        cv2.typing.MatLike: _description_
    """    
    screenshot = cv2.imread(file_path)

    screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    # Get the cell dimensions
    height, width = screenshot_gray.shape

    # Define the ROI (Region of Interest)
    # Cropping an area from the center
    start_x = (2 / 100 * width)
    start_y = (21 / 100 * height)
    end_x = (width - (.25 / 100 * width))
    end_y = (height - (35 / 100 * height))

    # Crop the image
    cropped_img = screenshot_gray[int(start_y):int(
        end_y), int(start_x):int(end_x)]

    return cropped_img


def generate_grid_image(puzzle: Puzzle) -> np.ndarray:
    """takes a Puzzle object as input and generates 
    an image of the puzzle grid. It draws the grid lines, 
    and for each cell, it draws the letter and score.

    Args:
        puzzle (Puzzle): the puzzle to generate the image for

    Returns:
        np.ndarray: the image of the puzzle grid
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


def draw_word_arrows(word_data: Tuple[str, int, List[Tuple[int, int]]], img: np.ndarray) -> np.ndarray:
    """ takes a tuple containing a word, its score, 
    and the path of cells it covers in the puzzle, 
    and an image of the puzzle grid. It draws arrows 
    on the image to represent the path of the word 
    in the puzzle.

    Args:
        word_data (Tuple[str, int, List[Tuple[int, int]]]): tuple containing a word, its score, and the path of cells it covers in the puzzle
        img (np.ndarray): image of the puzzle grid

    Returns:
        np.ndarray: the image of the puzzle grid with arrows drawn on it
    """    
    word, score, path = word_data
    arrow_color = (0, 0, 255)  # Blue color for the arrow
    arrow_thickness = 2

    for i in range(len(path) - 1):
        start_row, start_col = path[i]
        end_row, end_col = path[i + 1]

        start_point = (start_col * 70 + 35, start_row * 70 + 35)
        end_point = (end_col * 70 + 35, end_row * 70 + 35)

        cv2.arrowedLine(img, start_point, end_point, arrow_color, arrow_thickness)

    return img



def display_word_image(word_data: Tuple[str, int, List[Tuple[int, int]]], img: np.ndarray) -> None:
    """takes a tuple containing a word, its score, and the path of cells 
    it covers in the puzzle, and an image of the puzzle grid. 
    It creates a larger image with the grid image as the background, 
    adds the word and its score to the larger image, and displays the image.

    Args:
        word_data (Tuple[str, int, List[Tuple[int, int]]]): tuple containing a word, its score, and the path of cells it covers in the puzzle
        img (np.ndarray): image of the puzzle grid
    """    
    word, score, _ = word_data
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    line_type = cv2.LINE_AA

    # Calculate the text size and position for the word and score
    word_text_size, _ = cv2.getTextSize(word, font, font_scale, font_thickness)
    score_text_size, _ = cv2.getTextSize(str(score), font, font_scale, font_thickness)

    # Stack the word and score vertically with more spacing between them
    text_width = max(word_text_size[0], score_text_size[0])
    text_height = word_text_size[1] + score_text_size[1] + 30  # Increase the spacing between the word and score

    # Add spacing between the grid and both the word and score
    grid_spacing = 10
    text_height += grid_spacing
    text_width += grid_spacing

    # Create a larger image with the grid image as the background
    result_img = np.full((text_height + img.shape[0], text_width + img.shape[1], 3), 255, dtype=np.uint8)

    # Convert the img variable to a color image before assigning it to the result_img
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result_img[text_height:, :img.shape[1]] = img_color

    # Add the word and score to the larger image with the added spacing
    result_img = cv2.putText(result_img, word, (10 + grid_spacing, text_height - word_text_size[1] - 5), font, font_scale, (0, 0, 0), font_thickness, line_type)
    result_img = cv2.putText(result_img, f'Score: {score}', (10 + grid_spacing, text_height - 5), font, font_scale, (0, 0, 0), font_thickness, line_type)

    # Draw a border line around the entire grid (fix the offset)
    border_thickness = 2
    border_color = (0, 0, 0)
    top_left = (grid_spacing, text_height)
    bottom_right = (grid_spacing + img.shape[1] - 1, text_height + img.shape[0] - 1)
    cv2.rectangle(result_img, top_left, bottom_right, border_color, border_thickness)

    cv2.imshow('Word Image', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows


if __name__ == '__main__':
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
    print('Highest-scoring words:')
    for word, score, path in words:
        print(f'{word}: {score}')
    
    img = generate_grid_image(puzzle)
    
    for i, word_data in enumerate(words):
        word_img = img.copy()  # Create a copy of the original image for each word
        word_img = draw_word_arrows(word_data, word_img)

        display_word_image(word_data, word_img)
