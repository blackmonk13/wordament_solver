<p align="center">
  <a href="" rel="noopener">
 <img width=350px height=210px src="https://cdn.zone.msn.com/images/v9/en-us/game/mswm/350x210_mswm.png" alt="Project logo"></a>
</p>

<h3 align="center">Wordament Solver</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![GitHub Issues](https://img.shields.io/github/issues/blackmonk13/wordament_solver.svg)](https://github.com/blackmonk13/wordament_solver/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/blackmonk13/wordament_solver.svg)](https://github.com/blackmonk13/wordament_solver/pulls)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center"> A simple tool to help you find words in Wordament puzzles using image processing and a trie data structure.
    <br>
</p>

## 📝 Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Built Using](#built_using)
- [Authors](#authors)
- [Disclaimer](#disclaimer)

## 🧐 About <a name = "about"></a>

Wordament Solver is a Python project that helps you find words in Wordament puzzles. It uses image processing techniques to extract the grid data from a screenshot of the game, and then employs a trie data structure to efficiently search for valid words in the puzzle. The project also displays the highest-scoring words and shows the path of each word on the grid using arrows.

## 🏁 Getting Started <a name = "getting_started"></a>

To get started with Wordament Solver, follow these steps:

1. Clone the repository:
```
git clone https://github.com/blackmonk13/wordament_solver.git
```
2. Install the required packages:
```
pip install -r requirements.txt
```
3. Run the project using a sample puzzle or a screenshot of the game.

## 🎈 Usage <a name="usage"></a>

You can use Wordament Solver in two ways:

1. **Sample puzzle:** Define a sample puzzle using the `Cell` class and pass it to the `find_words` function. The function will return a list of highest-scoring words along with their scores and paths.

Example:
```python
if __name__ == '__main__':
    trie = load_word_list()
    cells = [
        [Cell('E', 2), Cell('S', 2), Cell('F', 5), Cell('E', 2)],
        [Cell('L', 3), Cell('A', 2), Cell('B', 5), Cell('D', 3)],
        [Cell('R', 2), Cell('S', 2), Cell('M', 4), Cell('G', 4)],
        [Cell('E', 2), Cell('I', 2), Cell('U', 4), Cell('E', 2)],
    ]
    puzzle = Puzzle(cells)
    words = find_words(puzzle, trie)

    # Print the highest-scoring words
    print('Highest-scoring words:')
    for word, score, path in words:
        print(f'{word}: {score}')
```

2. **Screenshot of the game:** Use the `get_latest_image` function to load a screenshot of the game, extract the grid data, and pass it to the `find_words` function. The function will return a list of highest-scoring words along with their scores and paths. Additionally, it will display an OpenCV image with the word score and a grid with arrows showing how to match the word.

Example:
```python
if __name__ == '__main__':
    latest_image = get_latest_image("data/")
    # Load the cropped image of the grid using OpenCV
    grid = get_grid(latest_image)

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
```

## ⛏️ Built Using <a name = "built_using"></a>

- [Python](https://www.python.org/) - Programming Language
- [NumPy](https://numpy.org/) - Array Processing Library
- [OpenCV](https://opencv.org/) - Image Processing Library
- [Pillow](https://pillow.readthedocs.io/en/stable/) - Image Processing Library
- [PyTesseract](https://github.com/madmaze/pytesseract) - OCR Library
- [NLTK](https://www.nltk.org/) - Natural Language Processing Library

## ✍️ Authors <a name = "authors"></a>

- [@blackmonk13](https://github.com/blackmonk13) - Idea & Initial work

See also the list of [contributors](https://github.com/blackmonk13/wordament_solver/contributors) who participated in this project.


## 📝 Disclaimer <a name = "disclaimer"></a>

Wordament Solver is not affiliated with, endorsed, or sponsored by Microsoft Corporation or the creators of Wordament. The project is intended for educational and entertainment purposes only. Any use of this project to gain an unfair advantage in the game or to violate the terms of service of Wordament or Microsoft Corporation is strictly prohibited. The authors and contributors of this project are not responsible for any misuse of the software.
