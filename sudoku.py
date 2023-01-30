import time

import numpy as np

##########################################################

class SudokuNode:
    def __init__(self, sudoku, valid_values, zero_position, bit_representation):
        self.sudoku = sudoku
        self.valid_values = valid_values
        self.zero_position = zero_position
        self.bit_representation = bit_representation

    def find_next_sudoku(self):  # takes the valid values for this node and applies the first one in the list to the
        # sudoku and returns the next sudoku and bit representation
        if len(self.valid_values) == 0:
            return None
        value = self.valid_values[0]
        del self.valid_values[0]
        next_sudoku = self.sudoku.copy()
        next_bit_rep = self.bit_representation.copy()
        next_sudoku[self.zero_position[0], self.zero_position[1]] = value
        next_bit_rep[0][self.zero_position[0]][value - 1] = 0
        next_bit_rep[1][self.zero_position[1]][value - 1] = 0
        block = 3 * (self.zero_position[0] // 3) + (self.zero_position[1] // 3)
        next_bit_rep[2][block][value - 1] = 0
        return next_sudoku, next_bit_rep


class ZerosDictionary:
    def __init__(self):
        self.dictionary = {}

    def next_zero(self):  # finds the empty cell stored in the dictionary with the least amount of valid values to go
        # in that cell
        min_values = 10
        key = ""
        for zero in self.dictionary:
            length = len(self.dictionary[zero][0])
            if length < min_values and self.dictionary[zero][2]:
                min_values = length
                key = zero
        try:
            self.dictionary[key][2] = False
        except KeyError:
            return -1
        return self.dictionary[key]


COL_ROW_START = [0, 0, 0, 3, 3, 3, 6, 6, 6]


def valid_values(position, bit_rep):  # calculates the valid values that could be put into an empty cell without
    # breaking any sudoku rules
    bit_row = bit_rep[0][position[0]]
    bit_col = bit_rep[1][position[1]]
    bit_block = bit_rep[2][3 * (position[0] // 3) + (position[1] // 3)]
    bit_valid_values = bit_block & bit_col & bit_row
    valid = []
    for i in range(9):
        if bit_valid_values[i] == 1:
            valid.append(i + 1)
    return valid


def filter_zeros(zeros, position):  # filters all empty cells from array that are not in the same row, column or 3x3
    # block as a particular position
    return np.array([zero for zero in zeros if zero[0] == position[0] or zero[1] == position[1] or (
                COL_ROW_START[zero[0]] == COL_ROW_START[position[0]] and COL_ROW_START[zero[1]] == COL_ROW_START[
            position[1]])])


def next_cell(sudoku, bit_rep, dictionary, position):  # re-calculates the valid values for empty cells that have
    # been affected by previously placing a value onto the sudoku. Updates the dictionary and uses the next_zero
    # function to return the next cell the program should explore
    if position is not None:
        zero_indices = filter_zeros(np.argwhere(sudoku == 0), position)
        num_zeros = zero_indices.shape[0]
        for i in range(num_zeros):
            zero_position = [zero_indices[i][0], zero_indices[i][1]]
            values = valid_values(zero_position, bit_rep)
            num_values = len(values)
            if num_values == 0:
                return 0
            dictionary.dictionary.update({str(zero_position[0]) + str(zero_position[1]): [values, zero_position, True]})
    return dictionary.next_zero()


def find_Three_By_Three(sudoku, position):  # takes a position and sudoku as input and returns a list that holds all
    # the values in the 3x3 block of the position
    row, column = position
    block_row = COL_ROW_START[row]
    block_column = COL_ROW_START[column]
    three_by_three = sudoku[block_row:block_row + 3, block_column:block_column + 3]
    return three_by_three.flatten()


def set_bit_for_index(indices):  # turns a list of numbers into a 9 bit number represented in a list. A 1 in the bit
    # list is at the index of a value in the indices list in the parameter
    bits = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in indices:
        bits[i - 1] = 1
    return bits


def find_bit_rep(sudoku):  # runs at the start of each puzzle. Uses the set_bit_for_index function to record which
    # values are in each row, column and 3x3 block. Also checks for invalid starting values in the puzzle e.g. Two 2s
    # in the top row.
    bit_representation = np.empty((3, 9, 9), int)  # [0] = rows, [1] = columns, [2] = blocks
    top_corners = [[0, 0], [0, 3], [0, 6], [3, 0], [3, 3], [3, 6], [6, 0], [6, 3], [6, 6]]
    ONE_TO_NINE = {1, 2, 3, 4, 5, 6, 7, 8, 9}
    for i in range(9):
        list_to_test = sudoku[i]
        if (np.unique(list_to_test[list_to_test != 0], return_counts=True)[
                1] > 1).any():  # https://stackoverflow.com/questions/5927180/how-do-i-remove-all-zero-elements-from-a-numpy-array
            # removes all 0s from list. then uses .unique() and tests whether the return counts array have any items
            # greater than 1
            return None
        bit_representation[0][i] = set_bit_for_index(ONE_TO_NINE.difference(set(sudoku[i])))
        list_to_test = sudoku[:, i]
        if (np.unique(list_to_test[list_to_test != 0], return_counts=True)[
                1] > 1).any():  # https://stackoverflow.com/questions/5927180/how-do-i-remove-all-zero-elements-from-a-numpy-array
            # removes all 0s from list. then uses .unique() and tests whether the return counts array have any items
            # greater than 1
            return None
        bit_representation[1][i] = set_bit_for_index(ONE_TO_NINE.difference(set(sudoku[:, i])))
        three_by_three = find_Three_By_Three(sudoku, top_corners[i])
        list_to_test = three_by_three
        if (np.unique(list_to_test[list_to_test != 0], return_counts=True)[
                1] > 1).any():  # https://stackoverflow.com/questions/5927180/how-do-i-remove-all-zero-elements-from-a-numpy-array
            # removes all 0s from list. then uses .unique() and tests whether the return counts array have any items
            # greater than 1
            return None
        bit_representation[2][i] = set_bit_for_index(ONE_TO_NINE.difference(set(three_by_three)))

    return bit_representation


def index_zeros(sudoku, bit_rep):  # looks at each empty cell and calculates the valid values that that cell can
    # take. This function is run at the start of solving each puzzle and when a backtrack is needed.
    dictionary = ZerosDictionary()
    zero_indices = np.argwhere(sudoku == 0)
    num_zeros = zero_indices.shape[0]
    if num_zeros == 0:  # puzzle complete
        return -1
    for i in range(num_zeros):
        zero_position = [zero_indices[i][0], zero_indices[i][1]]
        values = valid_values(zero_position, bit_rep)
        num_values = len(values)
        if num_values == 0:  # an empty cell has no valid values - backtrack
            return 0
        dictionary.dictionary[str(zero_position[0]) + str(zero_position[1])] = [values, zero_position, True]
    return dictionary


def sudoku_solver(sudoku):
    """
    Solves a Sudoku puzzle and returns its unique solution.

    Input
        sudoku : 9x9 numpy array
            Empty cells are designated by 0.

    Output
        9x9 numpy array of integers
            It contains the solution, if there is one. If there is no solution, all array entries should be -1.
    """
    bit_representation = find_bit_rep(sudoku)
    if bit_representation is None:
        return -1 * np.ones((9, 9))  # invalid starting values
    dictionary = index_zeros(sudoku, bit_representation)
    if dictionary == -1:  # puzzle complete
        return sudoku
    elif dictionary == 0:  # unsolvable
        return -1 * np.ones((9, 9))
    stack = []
    cell = next_cell(sudoku, bit_representation, dictionary, None)
    starting_node = SudokuNode(sudoku, cell[0], cell[1], bit_representation)
    stack.append(starting_node)
    index = False
    while len(stack) > 0:
        node = stack[-1]  # take top item from stack
        next_node_info = node.find_next_sudoku()
        if next_node_info is None:  # a position has no valid values - backtrack
            stack.pop()
            index = True
            continue
        next_sudoku = next_node_info[0]
        next_bit_rep = next_node_info[1]
        if index:
            index = False
            dictionary = index_zeros(next_sudoku, next_bit_rep)
        cell = next_cell(next_sudoku, next_bit_rep, dictionary, node.zero_position)
        if cell == -1:  # puzzle complete
            return next_sudoku
        elif cell == 0:  # next value
            continue
        next_node = SudokuNode(next_sudoku, cell[0], cell[1], next_bit_rep)
        stack.append(next_node)
    return -1 * np.ones((9, 9))  # stack empty, every node explored and no solution found


##################################################################################################

puzzle = np.array([[0, 8, 0, 0, 2, 0, 5, 6, 0],
                   [0, 0, 0, 1, 0, 0, 0, 0, 7],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 5, 0, 0, 9, 0, 4, 0, 8],
                   [0, 0, 7, 8, 0, 0, 0, 0, 3],
                   [0, 9, 0, 0, 1, 0, 0, 5, 0],
                   [2, 0, 4, 0, 0, 0, 8, 0, 0],
                   [0, 6, 0, 0, 8, 5, 0, 0, 0],
                   [0, 0, 0, 2, 0, 0, 1, 0, 0]])

start_time = time.process_time()
your_solution = sudoku_solver(puzzle)
end_time = time.process_time()
print(your_solution)
print(end_time - start_time)


SKIP_TESTS = False


def tests():
    import time
    difficulties = ['very_easy', 'easy', 'medium', 'hard']
    avr_time = 0
    for difficulty in difficulties:
        print(f"Testing {difficulty} sudokus")

        sudokus = np.load(f"data/{difficulty}_puzzle.npy")
        solutions = np.load(f"data/{difficulty}_solution.npy")

        count = 0
        for i in range(len(sudokus)):
            sudoku = sudokus[i].copy()
            print(f"This is {difficulty} sudoku number", i)
            print(sudoku)

            start_time = time.process_time()
            your_solution = sudoku_solver(sudoku)
            end_time = time.process_time()

            avr_time += (end_time - start_time)
            print(f"This is your solution for {difficulty} sudoku number", i)
            print(your_solution)

            print("Is your solution correct?")
            if np.array_equal(your_solution, solutions[i]):
                print("Yes! Correct solution.")
                count += 1
            else:
                print("No, the correct solution is:")
                print(solutions[i])

            print("This sudoku took", end_time - start_time, "seconds to solve.\n")

        print(f"{count}/{len(sudokus)} {difficulty} sudokus correct")
        if count < len(sudokus):
            break
    print(f"Average time per sudoku: {str(avr_time / 60.0)}")


if not SKIP_TESTS:
    tests()


def run_tests():
    difficulties = ['very_easy', 'easy', 'medium', 'hard']
    for difficulty in difficulties:
        sudokus = np.load(f"data/{difficulty}_puzzle.npy")
        start = time.perf_counter()
        for i in range(15):
            sudoku_solver(sudokus[i])
        end = time.perf_counter()

        print(f"{difficulty} average time: {(end - start) / 15:.10f}")


run_tests()
