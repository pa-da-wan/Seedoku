import numpy as np

class SudokuSolver:
    def __init__(self, sudoku_grid):
        self.sudoku = np.array(sudoku_grid)
    
    def findEmptyPos(self):
        # get the next row, col on the puzzle that's empty --> has the value 0
        # return (None, None) if there is none
        for row in range(9):
            for col in range(9): 
                if self.sudoku[row][col] == 0:
                    return row, col
        return None, None  

    def validityCheck(self, row, col, guess):
        # Check if the guessed number is valid in the row
        if guess in self.sudoku[row]:
            return False
        
        # Check if the guessed number is valid in the column
        if guess in self.sudoku[:, col]:
            return False

        # Check if the guessed number is valid in the 3x3 subgrid
        subgrid_row, subgrid_col = 3 * (row // 3), 3 * (col // 3)
        subgrid = self.sudoku[subgrid_row:subgrid_row+3, subgrid_col:subgrid_col+3]
        if guess in subgrid:
            return False

        return True

    def getSudokuSolution(self):
        empty_pos = self.findEmptyPos()

        # If there are no empty positions, the puzzle is solved
        if empty_pos is None:
            return self.sudoku.copy()  # Return a copy to avoid modifying the original puzzle

        row, col = empty_pos

        # Try numbers from 1 to 9
        for guess in range(1, 10):
            if self.validityCheck(row, col, guess):
                self.sudoku[row][col] = guess

                # Recursive call to solve the remaining puzzle
                solution = self.getSudokuSolution()
                if solution is not None:
                    return solution

                # If the current guess leads to an invalid solution, backtrack
                self.sudoku[row][col] = 0

        # No valid guess found
        return None
