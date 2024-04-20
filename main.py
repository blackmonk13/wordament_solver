import os
from typing import List
import cv2
import pytesseract
import math
import numpy as np

from utils import get_latest_image
from words_finder import find_words, load_word_list
from models import Cell, Puzzle


def get_score(cell: cv2.typing.MatLike) -> int:
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
    end_x = start_x + (77 / 100 * width)
    end_y = start_y + (92 / 100 * height)

    # Crop the image
    cropped_img = score_subcell[int(start_y):int(end_y), int(start_x):int(end_x)]

    # cv2.imshow('Cropped Image', cropped_img)
    # cv2.waitKey(0)

    # Rescale the image
    rescaled_image = cv2.resize(
        cropped_img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # Apply thresholding to the score subcell image
    _, score_thresh = cv2.threshold(
        rescaled_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Extract the score text from the thresholded image using pytesseract
    score_config = r'--psm 10 -c tessedit_char_whitelist=1234567890'
    score_text = pytesseract.image_to_string(score_thresh, config=score_config)
    score = int(score_text.strip()) if score_text.strip() else 1

    return score


def get_letter(cell: cv2.typing.MatLike) -> str:

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
    letter_config = r'--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ/-'
    letter_text = pytesseract.image_to_string(letter_bin, config=letter_config)
    letter = letter_text.strip().upper()

    return letter


def get_grid_data(grid: cv2.typing.MatLike) -> List[List[Cell]]:
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

def get_grid(filename:str) -> cv2.typing.MatLike:
    screenshot = cv2.imread(filename)
    
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
    cropped_img = screenshot_gray[int(start_y):int(end_y), int(start_x):int(end_x)]

    return cropped_img

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
    for word, score in words:
        print(f'{word}: {score}')
