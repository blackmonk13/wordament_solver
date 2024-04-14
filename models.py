from typing import List, Tuple, Dict

class Cell:
    def __init__(self, letter: str, score: int):
        self.letter = letter
        self.score = score

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
