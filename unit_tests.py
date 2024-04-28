import unittest

from wordament_solver.models import Cell, Puzzle
from wordament_solver.words_finder import find_words, load_word_list

class TestWordamentSolver(unittest.TestCase):
    def test_find_words(self):
        # create a sample puzzle and trie
        cells = [
            [Cell('E', 2), Cell('S', 2), Cell('F', 5), Cell('E', 2)],
            [Cell('L', 3), Cell('A', 2), Cell('B', 5), Cell('D', 3)],
            [Cell('R', 2), Cell('S', 2), Cell('M', 4), Cell('G', 4)],
            [Cell('E', 2), Cell('I', 2), Cell('U', 4), Cell('E', 2)],
        ]
        puzzle = Puzzle(cells)
        trie = load_word_list()

        # find the words in the puzzle
        words = find_words(puzzle, trie)

        # check that the words are correct
        self.assertIn(('BED', 10, ((1, 2), (0, 3), (1, 3))), words)
        self.assertIn(('EMBED', 16, ((3, 3), (2, 2), (1, 2), (0, 3), (1, 3))), words)
        self.assertIn(('DEFAME', 18, ((1, 3), (0, 3), (0, 2), (1, 1), (2, 2), (3, 3))), words)

if __name__ == '__main__':
    unittest.main()
