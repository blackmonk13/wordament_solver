import nltk
from nltk.corpus import words
from typing import List, Set, Tuple
from .models import Puzzle, Cell, Trie

DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0),
              (1, 1), (1, -1), (-1, 1), (-1, -1)]


def find_words(puzzle: Puzzle, trie: Trie) -> List[Tuple[str, int, List[Tuple[int, int]]]]:
    """finds all the words in the given puzzle that are present in the Trie. 
    It uses a DFS algorithm to explore all possible paths in the puzzle. 
    For each path, it checks if the formed word is present in the Trie and 
    if it meets certain conditions (length >= 3, starts with prefix, ends with suffix). 
    If so, it calculates the score of the path and adds the word to 
    the results. The results are then sorted by score in descending order.

    Args:
        puzzle (Puzzle): the puzzle to find words in
        trie (Trie): the Trie containing the words

    Returns:
        List[Tuple[str, int, List[Tuple[int, int]]]]: List of tuples containing the word, score and path
    """
    def dfs(row: int, col: int, word: str, visited: Set[Tuple[int, int]], prefix: str, suffix: str, path: List[Tuple[int, int]]) -> None:
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
                    # Get the score based on the path
                    score = puzzle.get_score(path)

                    # Only add the word if it's not already in the results
                    # this ensures only one instance of every word
                    if word not in results:
                        results[word] = (score, tuple(path))

                for dr, dc in DIRECTIONS:
                    r, c = row + dr, col + dc
                    if (
                        0 <= r < puzzle.nrows
                        and 0 <= c < puzzle.ncols
                        and puzzle[r, c].letter != ''
                    ):
                        path.append((r, c))  # Add the cell to the path
                        dfs(r, c, word, visited, prefix, suffix, path)
                        path.pop()  # Remove the cell from the path

                word = word[:-len(option)]

        visited.remove((row, col))

    results = {}
    for row in range(puzzle.nrows):
        for col in range(puzzle.ncols):
            dfs(row, col, '', set(), '', '', [(row, col)])

    # Sort by score
    return sorted([(word, score, path) for word, (score, path) in results.items()], key=lambda x: x[1], reverse=True)


def load_word_list() -> Trie:
    """loads a list of words from the nltk corpus, 
    inserts each word into a Trie, and returns the Trie.

    Returns:
        Trie: the Trie containing the words
    """    
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

    # Print the highest-scoring words
    print('Highest-scoring words:')
    for word, score, path in words:
        print(f'{word}: {score}')
