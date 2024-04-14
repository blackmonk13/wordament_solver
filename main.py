from typing import List, Set, Tuple
from models import Puzzle, Cell

DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0),
              (1, 1), (1, -1), (-1, 1), (-1, -1)]


def find_words(puzzle: Puzzle, word_list: Set[str]) -> List[str]:
    def dfs(row: int, col: int, word: str, visited: Set[Tuple[int, int]], prefix: str, suffix: str) -> None:
        cell = puzzle[row, col]
        if cell.letter == '':
            return
        if (row, col) in visited:
            return
        visited.add((row, col))

        if cell.prefix:
            prefix = cell.letter
        elif cell.suffix:
            suffix = cell.letter
        else:
            for option in cell.options:
                word += option
                if word in word_list and len(word) >= 3 and word.startswith(prefix) and word.endswith(suffix):
                    results.add((word, puzzle.get_score(word)))

                for dr, dc in DIRECTIONS:
                    r, c = row + dr, col + dc
                    if (
                        0 <= r < puzzle.nrows
                        and 0 <= c < puzzle.ncols
                        and puzzle[r, c].letter != ''
                    ):
                        dfs(r, c, word, visited, prefix, suffix)

                word = word[:-len(option)]

        visited.remove((row, col))

    results = set()
    for row in range(puzzle.nrows):
        for col in range(puzzle.ncols):
            dfs(row, col, '', set(), '', '')
    return [word for word, score in results]


def load_word_list(filename: str) -> Set[str]:
    with open(filename, 'r') as file:
        return {word.strip().upper() for word in file.read().split()}


if __name__ == '__main__':
    word_list = load_word_list('data/words_alpha.txt')
    cells = [
        [Cell('T', 2), Cell('R', 2), Cell('A', 2), Cell('S', 2)],
        [Cell('N', 2), Cell('O', 2), Cell('G', 4), Cell('G', 4)],
        [Cell('O', 2), Cell('U/M', 20), Cell('L', 3), Cell('A', 2)],
        [Cell('W', 6), Cell('R', 2), Cell('O', 2), Cell('P', 4)],
    ]
    puzzle = Puzzle(cells)
    words = find_words(puzzle, word_list)
    print(words)
