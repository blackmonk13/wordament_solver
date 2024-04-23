from typing import List, Tuple, Dict

class TrieNode:
    def __init__(self):
        self.children = {}
        self.end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.end_of_word = True

    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.end_of_word

class Cell:
    def __init__(self, letter: str, score: int):
        self.letter = letter
        self.score = score
        self.prefix = letter.startswith('-')
        self.suffix = letter.endswith('-')
        if self.prefix:
            self.letter = self.letter[1:]
        if self.suffix:
            self.letter = self.letter[:-1]
        self.options = letter.split('/')
        if len(self.options) == 1:
            self.options = self.options[0]

    def matches(self, letter: str) -> bool:
        if isinstance(self.options, str):
            return self.options == letter
        else:
            return letter in self.options
        
    def __str__(self) -> str:
        result = ''
        result += ' '.join(f'{self.letter}({self.score})')
        return result.strip()


class Puzzle:
    def __init__(self, cells: List[List[Cell]]):
        self.cells = cells
        self.nrows = len(cells)
        self.ncols = len(cells[0]) if cells else 0

    def __getitem__(self, index: Tuple[int, int]) -> Cell:
        row, col = index
        return self.cells[row][col]

    def __str__(self) -> str:
        result = ''
        for row in self.cells:
            result += ' '.join(f'{cell.letter}({cell.score})' for cell in row) + '\n'
        return result.strip()
    
    def get_score(self, path: List[Tuple[int, int]]) -> int:
        score = 0
        for row, col in path:
            cell = self.cells[row][col]
            score += cell.score
        return score


if __name__ == '__main__':
    cells = [
        [Cell('A', 1), Cell('B', 1), Cell('C', 1), Cell('D', 1)],
        [Cell('E', 1), Cell('F', 1), Cell('G', 1), Cell('H', 1)],
        [Cell('I', 1), Cell('J', 1), Cell('K', 1), Cell('L', 1)],
        [Cell('M', 1), Cell('N', 1), Cell('O', 1), Cell('P', 1)],
    ]
    puzzle = Puzzle(cells)
    print(puzzle)
