import torch
import numpy as np
import random
from torch.utils.data import Dataset

def is_valid(board, row, col, num):
    """ check if placing num at (row, col) is valid"""
    if num in board[row]: # check rows
        return False

    if num in board[:, col]: # check columns
        return False

    box_row, box_col = 3 * (row // 3), 3 * (col // 3)   # Check 3x3 box
    if num in board[box_row:box_row+3, box_col:box_col+3]:
        return False

    return True

def solve_sudoku(board):
    """ Fill empty cells (0) with backtracking """
    for row in range(9):
        for col in range(9):
            if board[row, col] == 0:
                for num in range(1, 10):
                    if is_valid(board, row, col, num):
                        board[row, col] = num

                        if solve_sudoku(board):
                            return True

                        board[row, col] = 0

                return False

    return True

def generate_valid_sudoku():
    """ generate a complete and valid sudoku puzzle """
    board = np.zeros((9, 9), dtype=int)

    # fill main diagonal with random numbers
    for i in range(3):
        nums = list(range(1, 10))
        random.shuffle(nums)
        for j in range(3):
            board[i*3 + j, i*3 + j] = nums[j]

    # solve the rest with backtracking
    solve_sudoku(board)

    return board.flatten().astype(int)  # Return as 1D (81 elements)

class SudokuDataset(Dataset):
    def __init__(self, num_samples=100):
        """ create sudoku puzzles with multiple cells masked """
        self.puzzles = []
        self.solutions = []

        print(f" generating {num_samples} sudoku puzzles...")

        for i in range(num_samples):
            # generate complete sudoku
            solution = generate_valid_sudoku()
            
            # reate puzzle by masking ~40 cells randomly
            puzzle = solution.copy()
            num_masked = random.randint(35, 45)
            masked_indices = random.sample(range(81), num_masked)
            puzzle[masked_indices] = 0
            
            self.puzzles.append(puzzle)
            self.solutions.append(solution) 
            
            if (i + 1) % 20 == 0:
                print(f"  Generated {i + 1}/{num_samples}")

        print("Dataset generation complete")

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        puzzle = self.puzzles[idx]
        solution = self.solutions[idx]

        return(
            torch.tensor(puzzle, dtype=torch.long),
            torch.tensor(solution, dtype=torch.long) 
        )

