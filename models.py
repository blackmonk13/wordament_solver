from typing import List, Tuple, Dict

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
    
    def get_score(self, word: str) -> int:
        score = 0
        for letter in word:
            for row in range(self.nrows):
                for col in range(self.ncols):
                    cell = self.cells[row][col]
                    if cell.letter == letter:
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
