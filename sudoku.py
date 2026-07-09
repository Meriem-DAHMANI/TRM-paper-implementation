import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from trm import create_trm_mlp
from train import train_step_with_clipping
from evaluate import evaluate_accuracy

class SudokuDataset(Dataset):
    """
    Dataset for Sudoku puzzles.
    
    Input: 9x9 grid with some numbers filled in
    Output: Complete 9x9 grid
    """
    
    def __init__(self, puzzles, solutions):
        """
        puzzles: List of 9x9 numpy arrays (0 = empty cell)
        solutions: List of 9x9 numpy arrays (complete grids)
        """
        self.puzzles = puzzles
        self.solutions = solutions
    
    def __len__(self):
        return len(self.puzzles)
    
    def __getitem__(self, idx):
        # Flatten grid to sequence
        puzzle = self.puzzles[idx].flatten()  # 81 tokens
        solution = self.solutions[idx].flatten()  # 81 tokens
        
        # Add 1 to avoid 0 (reserved for padding)
        puzzle = torch.tensor(puzzle + 1, dtype=torch.long)
        solution = torch.tensor(solution + 1, dtype=torch.long)
        
        return puzzle, solution


def train_sudoku_solver():
    """
    Train TRM to solve Sudoku puzzles.
    
    Paper results:
    - TRM-MLP: 87.4% accuracy on Sudoku-Extreme
    - DeepSeek R1 (671B params): 0.0% accuracy
    - Claude 3.7: 0.0% accuracy
    """
    print(" Training Sudoku Solver")
    print("=" * 50)
    
    # Model setup
    vocab_size = 11  # Digits 1-9, plus 0 for padding, plus 1 for BOS/EOS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use TRM-MLP (better for Sudoku!)
    model = create_trm_mlp(vocab_size=vocab_size, d_model=256, n_layers=4)
    model = model.to(device)
    
    print(f"Model: TRM-MLP")
    print(f"Parameters: {model.count_parameters() / 1e6:.2f}M")
    print(f"Device: {device}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Load Sudoku data (you'd replace this with actual data)
    # puzzles = load_sudoku_puzzles()
    # solutions = load_sudoku_solutions()
    # 
    # train_dataset = SudokuDataset(puzzles, solutions)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Training loop
    # num_epochs = 100
    # for epoch in range(num_epochs):
    #     epoch_loss = 0
    #     for puzzles, solutions in train_loader:
    #         puzzles = puzzles.to(device)
    #         solutions = solutions.to(device)
    #         
    #         loss = train_step_with_clipping(
    #             model, puzzles, solutions, optimizer, criterion
    #         )
    #         epoch_loss += loss
    #     
    #     avg_loss = epoch_loss / len(train_loader)
    #     
    #     if (epoch + 1) % 10 == 0:
    #         # Evaluate
    #         acc = evaluate_accuracy(model, val_loader, device)
    #         print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")
    
    return model


def solve_sudoku_puzzle(model, puzzle, device):
    """
    Solve a single Sudoku puzzle.
    
    Args:
        model: Trained TRM model
        puzzle: 9x9 numpy array (0 = empty)
        device: torch device
    
    Returns:
        solution: 9x9 numpy array (complete grid)
    """
    model.eval()
    
    # Flatten and convert to tensor
    puzzle_flat = torch.tensor(puzzle.flatten() + 1, dtype=torch.long)
    puzzle_flat = puzzle_flat.unsqueeze(0).to(device)
    
    # Generate solution
    with torch.no_grad():
        solution_flat = model.generate(
            puzzle_flat,
            max_length=81,
            latent_len=32,
            temperature=0.1  # Low temperature for deterministic solving
        )
    
    # Convert back to 9x9 grid
    solution = solution_flat[0].cpu().numpy() - 1
    solution = solution.reshape(9, 9)
    
    return solution


# Example usage
if __name__ == "__main__":
    # Create and train model
    # model = train_sudoku_solver()
    
    # Solve a puzzle
    # puzzle = np.array([
    #     [5, 3, 0, 0, 7, 0, 0, 0, 0],
    #     [6, 0, 0, 1, 9, 5, 0, 0, 0],
    #     ...
    # ])
    # 
    # solution = solve_sudoku_puzzle(model, puzzle, device)
    # print("Solution:")
    # print(solution)
    pass