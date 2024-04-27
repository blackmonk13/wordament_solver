from .image_parser import display_word_image, draw_word_arrows, generate_grid_image, generate_word_image, get_grid, get_grid_data
from .models import Puzzle
from .utils import get_latest_image
from .words_finder import find_words, load_word_list


if __name__ == '__main__':
    print('Wordament Solver')
    latest_image = get_latest_image('data/')

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
        word_img = generate_word_image(word_data, word_img)

        display_word_image(word_data, word_img)