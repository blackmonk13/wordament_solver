from typing import List, Set, Tuple
from models import Puzzle, Cell

DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def find_words(puzzle: Puzzle, word_list: Set[str]) -> List[str]:
    def dfs(row: int, col: int, word: str) -> None:
        cell = puzzle[row, col]
        word += cell.letter
        if word not in word_set:
            return
        
        if word in word_list and len(word) > 2:
            # print(f'{word} - length {len(word)}')
            results.add(word)
        for dr, dc in DIRECTIONS:
            r, c = row + dr, col + dc
            if 0 <= r < puzzle.nrows and 0 <= c < puzzle.ncols:
                dfs(r, c, word)
        
        if len(word) > 1:
            word_set.discard(word)

    word_set = set(word_list)
    word_set.update([word[:i] for word in word_list for i in range(1, len(word))])
    results = set()
    for row in range(puzzle.nrows):
        for col in range(puzzle.ncols):
            dfs(row, col, '')
    return list(results)

def load_word_list(filename: str) -> Set[str]:
    with open(filename, 'r') as file:
        return {word.strip().upper() for word in file.read().split()}
    
if __name__ == '__main__':
    word_list = load_word_list('data/words_alpha.txt')
    cells = [
        [Cell('G', 4), Cell('B', 5), Cell('P', 4), Cell('I', 2)],
        [Cell('O', 2), Cell('S', 2), Cell('L', 3), Cell('N', 2)],
        [Cell('P', 4), Cell('S', 2), Cell('R', 2), Cell('E', 1)],
        [Cell('O', 2), Cell('A', 2), Cell('A', 2), Cell('E', 1)],
    ]
    puzzle = Puzzle(cells)
    words = find_words(puzzle, word_list)
    print(words)