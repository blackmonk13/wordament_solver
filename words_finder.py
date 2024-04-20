import nltk
from nltk.corpus import words
from typing import List, Set, Tuple
from models import Puzzle, Cell, Trie

DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0),
              (1, 1), (1, -1), (-1, 1), (-1, -1)]


def find_words(puzzle: Puzzle, trie: Trie) -> List[Tuple[str, int]]:
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
                if trie.search(word) and len(word) >= 3 and word.startswith(prefix) and word.endswith(suffix):
                    score = puzzle.get_score(word)
                    results.add((word, score))

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
    return sorted(list(results), key=lambda x: x[1], reverse=True)


def load_word_list() -> Trie:
    trie = Trie()
    nltk.download('words')
    word_list = words.words()
    for word in word_list:
        if len(word) >= 3:
            trie.insert(word.strip().upper())
    return trie


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
    print(words)

