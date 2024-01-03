class SudokuSolver:
    def __init__(self, sudoku_grid):
        self.sudoku = sudoku_grid

    def findEmptyPos(self):
        for row in range(9):
            for col in range(9):
                if self.sudoku[row][col] == 0:
                    return row, col
        return None, None

    def isValidMove(self, row, col, num):
        # Check if the number is not present in the row, column, and subgrid
        return (
            all(num != self.sudoku[row][j] for j in range(9)) and
            all(num != self.sudoku[i][col] for i in range(9)) and
            all(num != self.sudoku[row - row % 3 + i][col - col % 3 + j] for i in range(3) for j in range(3))
        )

    def solveSudoku(self):
        row, col = self.findEmptyPos()

        if row is None:
            return True  # Sudoku is solved

        for num in range(1, 10):
            if self.isValidMove(row, col, num):
                self.sudoku[row][col] = num

                if self.solveSudoku():
                    return True  # Solution found

                self.sudoku[row][col] = 0  # Backtrack if the solution is not found

        return False  # No solution found

    def propagateConstraints(self):
        for row in range(9):
            for col in range(9):
                if self.sudoku[row][col] != 0:
                    num = self.sudoku[row][col]

                    # Clear possibilities in the same row and column
                    for i in range(9):
                        if self.sudoku[row][i] == 0:
                            self.sudoku[row][i] = -num
                        if self.sudoku[i][col] == 0:
                            self.sudoku[i][col] = -num

                    # Clear possibilities in the same subgrid
                    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
                    for i in range(start_row, start_row + 3):
                        for j in range(start_col, start_col + 3):
                            if self.sudoku[i][j] == 0:
                                self.sudoku[i][j] = -num

    def optimizeSolveSudoku(self):
        self.solveSudoku()
        self.propagateConstraints()

    def getSudokuSolution(self):
        sudokuCopy = [row[:] for row in self.sudoku]

        self.optimizeSolveSudoku()

        return self.sudoku if any(0 in row for row in self.sudoku) else sudokuCopy
