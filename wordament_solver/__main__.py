import argparse
import csv
import json
import os

import cv2
from .image_parser import display_word_image, draw_word_arrows, generate_grid_image, generate_word_image, get_grid, get_grid_data
from .models import Puzzle
from .utils import get_latest_image
from .words_finder import find_words, load_word_list


def main():
    parser = argparse.ArgumentParser(
        prog="wordament_solver",
        description='A simple tool to help you find words in Wordament puzzles using image processing and a trie data structure.')

    # Add an argument for the input method
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--screenshot', type=str,
                             help='Path to a screenshot of the puzzle grid')
    input_group.add_argument(
        '--grid', type=str, help='Raw grid data in a specific format')

    # Add an argument for the cropped image
    parser.add_argument('--cropped', action='store_true',
                        help='Whether the input image is already cropped or not')

    # Add an argument for the output method
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument(
        '--print', action='store_true', help='Print the results to the console')
    output_group.add_argument(
        '--json', type=str, help='Path to a JSON file to write the results to')
    output_group.add_argument(
        '--csv', type=str, help='Path to a CSV file to write the results to')

    # Add an argument for the output format
    parser.add_argument('--format', choices=['words', 'words_scores',
                        'words_scores_paths'], default='words_scores', help='Format of the output')

    # Add an argument for the output image
    parser.add_argument('--image', action='store_true',
                        help='Display the solved word image')
    parser.add_argument('--output_dir', type=str,
                        help='Directory to write the solved word images to')
    
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')


    args = parser.parse_args()

    # Process the input and output arguments
    if args.screenshot:
        if not os.path.exists(args.screenshot):
            raise FileNotFoundError(f"The file '{args.screenshot}' does not exist.")
        # Load the puzzle grid from the screenshot
        grid = get_grid(args.screenshot, args.cropped)
        cells = get_grid_data(grid)
        puzzle = Puzzle(cells)
    elif args.grid:
        # Parse the raw grid data
        pass

    # Load the word list
    trie = load_word_list()

    # Solve the puzzle
    words = find_words(puzzle, trie)

    # Format the output
    if args.format == 'words':
        output = [word for word, score, path in words]
    elif args.format == 'words_scores':
        output = [(word, score) for word, score, path in words]
    else:
        output = words

    # Print or write the output
    if args.print:
        print(output)
    elif args.json:
        if not os.path.exists(args.json):
            raise FileNotFoundError(f"The file '{args.json}' does not exist.")
        # Write the output to a JSON file
        with open(args.json, 'w') as f:
            json.dump(output, f)
    else:
        if not os.path.exists(args.csv):
            raise FileNotFoundError(f"The file '{args.csv}' does not exist.")
        with open(args.csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(output)

    if not args.image or not args.output_dir:
        return

    # Generate the grid image
    img = generate_grid_image(puzzle)

    for i, word_data in enumerate(words):
        word_img = img.copy()  # Create a copy of the original image for each word
        word_img = draw_word_arrows(word_data, word_img)
        word_img = generate_word_image(word_data, word_img)

        if args.image:
            display_word_image(word_data, word_img)
        elif args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(args.output_dir,
                        f"{word_data[0]}_{word_data[1]}.png"), word_img)


if __name__ == '__main__':
    main()
