import torch
import numpy as np
import random
from torch.utils.data import Dataset

def is_valid(board, row, col, num):
    """Check if placing num at (row, col) is valid"""
    if num in board[row]:  # check rows
        return False

    if num in board[:, col]:  # check columns
        return False

    box_row, box_col = 3 * (row // 3), 3 * (col // 3)  # Check 3x3 box
    if num in board[box_row:box_row+3, box_col:box_col+3]:
        return False

    return True


def solve_sudoku(board):
    """Solve sudoku and return solved board"""
    board = board.copy()
    
    for row in range(9):
        for col in range(9):
            if board[row, col] == 0:
                for num in range(1, 10):
                    if is_valid(board, row, col, num):
                        board[row, col] = num
                        if solve_sudoku(board) is not None:
                            return board
                        board[row, col] = 0
                return None
    return board


def get_difficulty_config(board_size=9):
    """Get difficulty config for given board size"""
    total_cells = board_size * board_size
    
    difficulty_config = {
        "easy": (int(total_cells * 0.25), int(total_cells * 0.35)),
        "medium": (int(total_cells * 0.40), int(total_cells * 0.55)),
        "hard": (int(total_cells * 0.60), int(total_cells * 0.75)),
    }
    return difficulty_config


def generate_sudoku(board_size=9, difficulty="medium"):
    """Generate a sudoku puzzle"""
    
    difficulty_config = get_difficulty_config(board_size)
    min_masked, max_masked = difficulty_config[difficulty]
    
    # Generate solved sudoku
    board = np.zeros((board_size, board_size), dtype=np.int32)
    
    # Fill diagonal 3x3 boxes
    box_size = 3
    for box in range(box_size):
        nums = list(range(1, board_size + 1))
        random.shuffle(nums)
        for i in range(box_size):
            for j in range(box_size):
                board[box*box_size + i, box*box_size + j] = nums[i*box_size + j]
    
    # Solve the rest
    for row in range(board_size):
        for col in range(board_size):
            if board[row, col] == 0:
                for num in range(1, board_size + 1):
                    if is_valid(board, row, col, num):
                        board[row, col] = num
                        if solve_sudoku(board) is not None:
                            break
    
    solved_board = board.copy()
    
    # Create puzzle by masking cells
    num_masked = random.randint(min_masked, max_masked)
    masked_positions = random.sample(range(board_size * board_size), num_masked)
    
    puzzle = board.copy()
    for pos in masked_positions:
        puzzle[pos // board_size, pos % board_size] = 0
    
    return puzzle, solved_board


class SudokuDataset(Dataset):
    def __init__(self, num_samples=1000, board_size=9, difficulty="medium"):
        self.num_samples = num_samples
        self.board_size = board_size
        self.difficulty = difficulty
        self.puzzles = []
        self.solutions = []
        
        print(f"Generating {num_samples} sudoku puzzles ({board_size}x{board_size}, {difficulty})...")
        for _ in range(num_samples):
            puzzle, solution = generate_sudoku(board_size, difficulty)
            self.puzzles.append(puzzle)
            self.solutions.append(solution)
        print(f"Generated {num_samples} puzzles!")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        puzzle = torch.tensor(self.puzzles[idx], dtype=torch.float32).unsqueeze(0)  # (1, 9, 9)
        solution = torch.tensor(self.solutions[idx], dtype=torch.float32).unsqueeze(0)  # (1, 9, 9)
        
        return puzzle, solution