import nltk
from nltk.corpus import words
from typing import List, Set, Tuple
from .models import Puzzle, Cell, Trie

DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def load_word_list() -> Trie:
    """loads a list of words from the nltk corpus,
    inserts each word into a Trie, and returns the Trie.

    Returns:
        Trie: the Trie containing the words
    """
    trie = Trie()
    nltk.download("words")
    word_list = words.words()
    for word in word_list:
        if len(word) >= 3:
            trie.insert(word.strip().upper())
    return trie


def depth_first_search(
    puzzle: Puzzle,
    trie: Trie,
    row: int,
    col: int,
    prefix: str,
    visited: Set[Tuple[int, int]],
    path: List[Tuple[int, int]],
) -> List[Tuple[str, int, List[Tuple[int, int]]]]:
    """
    Finds valid words in the puzzle using a depth-first search (DFS) approach.

    Args:
        puzzle (Puzzle): The puzzle grid.
        trie (Trie): The trie containing valid words.
        row (int): Current row index.
        col (int): Current column index.
        prefix (str): Current word prefix.
        visited (Set[Tuple[int, int]]): Set of visited cell coordinates.
        path (List[Tuple[int, int]]): Current path.

    Returns:
        List[Tuple[str, int, List[Tuple[int, int]]]]:
            List of valid words with scores and paths.
            Each tuple contains:
                - The word (e.g., "HERO")
                - The word's score (e.g., 12)
                - The path taken to create the word (a list of cell coordinates)
    """
    if (
        row < 0
        or row >= puzzle.nrows
        or col < 0
        or col >= puzzle.ncols
        or (row, col) in visited
    ):
        return []

    cell = puzzle[row, col]
    options = cell.options if isinstance(cell.options, list) else [cell.options]

    words = []
    for option in options:
        if not cell.matches(option):
            continue

        # Check if the current letter matches the prefix of the current cell
        if cell.prefix and prefix and not prefix.startswith(option):
            continue

        # Add the letter to the word
        new_prefix = prefix + option

        node = trie.root
        for char in new_prefix:
            if char not in node.children:
                break
            node = node.children[char]
        else:
            if node.end_of_word:
                word_score = puzzle.get_score(path + [(row, col)])
                words.append((new_prefix, word_score, path + [(row, col)]))

            visited.add((row, col))
            for dr, dc in DIRECTIONS:
                words.extend(
                    depth_first_search(
                        puzzle,
                        trie,
                        row + dr,
                        col + dc,
                        new_prefix,
                        visited,
                        path + [(row, col)],
                    )
                )
            visited.remove((row, col))

    return words


def find_words(
    puzzle: Puzzle, trie: Trie, sort: bool = False
) -> List[Tuple[str, int, List[Tuple[int, int]]]]:
    """
    Finds valid words in the puzzle and optionally sorts them by score.

    Args:
        puzzle (Puzzle): The puzzle grid.
        trie (Trie): The trie containing valid words.
        sort (bool, optional): Whether to sort words by score (default is False).

    Returns:
        List[Tuple[str, int, List[Tuple[int, int]]]]:
            List of valid words with scores and paths.
            Each tuple contains:
                - The word (e.g., "HERO")
                - The word's score (e.g., 12)
                - The path taken to create the word (a list of cell coordinates)
    """
    # Call find_words for each cell
    all_words = []
    for row in range(puzzle.nrows):
        for col in range(puzzle.ncols):
            words_with_path = depth_first_search(puzzle, trie, row, col, "", set(), [])
            all_words.extend(words_with_path)

    if not sort:
        return all_words
    # Sort the words by their score (highest to lowest)
    sorted_words = sorted(all_words, key=lambda x: x[1], reverse=True)

    return sorted_words


if __name__ == "__main__":
    trie = load_word_list()
    cells = [
        [Cell("E", 2), Cell("S", 2), Cell("F", 5), Cell("E", 2)],
        [Cell("L", 3), Cell("A", 2), Cell("B", 5), Cell("D", 3)],
        [Cell("R", 2), Cell("S", 2), Cell("M", 4), Cell("G", 4)],
        [Cell("E", 2), Cell("I", 2), Cell("U", 4), Cell("E", 2)],
    ]
    puzzle = Puzzle(cells)
    # Solve the puzzle
    words = find_words(puzzle, trie)

    # Print the sorted words
    for word, score, path in words:
        print(f"Word: {word}, Score: {score}, Path: {path}")
